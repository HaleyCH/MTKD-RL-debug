from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, DistillKL, correct_num
import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np
from distiller_zoo import FeatureKLLoss, FeatureMSELoss


def get_actions(agent_pred):
    batch_size = agent_pred.size(0)
    teacher_num = agent_pred.size(1)
    index = torch.from_numpy(np.random.randint(0, teacher_num, batch_size).astype(np.int64)).cuda(args.gpu)
    random_select =  F.one_hot(index, num_classes=teacher_num).float().cuda(args.gpu)
    actions = torch.where(agent_pred>=0.5, torch.ones_like(agent_pred), torch.zeros_like(agent_pred)).cuda(args.gpu)
    is_random = (actions.sum(1) ==  0)[:, None].float().cuda(args.gpu)
    actions = actions + random_select * is_random
    return actions


def train_agent(args, epoch, agent_state, agent_rewards, logits_agent_actions, agent, agent_optimizer):
    agent.train()
    agent_loss = AverageMeter('agent_loss', ':.4e')

    for state, rewards, actions in zip(agent_state, agent_rewards, logits_agent_actions):
        
        agent_pred = agent(state)
        agent_optimizer.zero_grad() 
        
        action_label = torch.ones_like(agent_pred[0]).detach()
        loss_logits = F.binary_cross_entropy(agent_pred[0], action_label, weight=rewards.unsqueeze(-1))
        loss_feature = F.binary_cross_entropy(agent_pred[1], action_label, weight=rewards.unsqueeze(-1))
        loss = loss_feature + loss_logits
        loss.backward()
        agent_optimizer.step()

        agent_loss.update(loss.item(), actions.size(0))

    if args.rank == 0:
        args.logger.info('Epoch:{}, agent Loss:{:.6f}'.format(epoch, agent_loss.avg))

def get_agent_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div):
    trans_student_embeddings = []
    for idx in range(len(trans_student_features)):
        trans_student_embedding = F.adaptive_avg_pool2d(trans_student_features[idx], (1,1))
        trans_student_embedding = trans_student_embedding.view(trans_student_embedding.size(0), -1)
        trans_student_embeddings.append(trans_student_embedding)

    teacher_infos = []
    t_ces = []
    t_s_feat_div = []
    t_s_logit_div = []
    for idx in range(len(teacher_embeddings)):            
        feat_cos_sim = F.cosine_similarity(trans_student_embeddings[idx], teacher_embeddings[idx]).unsqueeze(-1)
        t_s_feat_div.append(feat_cos_sim)
        logit_kl = criterion_div(logits, teacher_logits[idx], unreduce=True).unsqueeze(-1)
        t_s_logit_div.append(logit_kl)
        teachers_ce = F.cross_entropy(teacher_logits[idx], targets, reduction='none').unsqueeze(-1)
        t_ces.append(teachers_ce)
        teacher_info = torch.cat([feat_cos_sim, logit_kl, teachers_ce, teacher_embeddings[idx], teacher_logits[idx]], dim=1).detach()
        teacher_infos.append(teacher_info)
    t_ces = torch.cat(t_ces, dim=1).detach()
    t_s_logit_div = torch.cat(t_s_logit_div, dim=1).detach()
    t_s_feat_div = torch.cat(t_s_feat_div, dim=1).detach()
    return teacher_infos, t_ces, t_s_logit_div, t_s_feat_div


def train_avg(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, feat_trans, teacher_models):
    
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_kd = AverageMeter('train_loss_kd', ':.4e')
    train_loss_feat = AverageMeter('train_loss_feat', ':.4e')

    top1_num = 0
    top5_num = 0
    total = 0

    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start_time = time.time()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        features, logits = model(inputs, is_feat=True) 
        trans_student_features = feat_trans(features[-2])
        
        teacher_logits = []
        teacher_features = []
        teacher_embeddings = []
        with torch.no_grad():
            all_teacher_info = []
            for t_model in teacher_models:
                t_features, t_logits = t_model(inputs, is_feat=True)
                t_feature = t_features[-1]
                t_feature = t_feature.detach() 
                t_logits = t_logits.detach() 
                
                teacher_features.append(t_features[-2])
                teacher_logits.append(t_logits)
                teacher_embeddings.append(t_features[-1])

        loss_cls = criterion_ce(logits, targets)
        
        loss_kd = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(teacher_models)):
            loss_kd = loss_kd + criterion_div(logits, teacher_logits[idx].detach())
        loss_kd = loss_kd / len(teacher_models)
        loss_feat = torch.tensor(0.).cuda(args.gpu)
        
        if args.feat_kd == 'mse':
            feat_kd_func = FeatureMSELoss()
        elif args.feat_kd == 'kl':
            feat_kd_func = FeatureKLLoss(args.kd_T)

        for idx in range(len(teacher_models)):
            loss_feat = loss_feat +  (feat_kd_func(trans_student_features[idx], teacher_features[idx])).mean()
        loss_feat = loss_feat / len(teacher_models)
        loss_feat = args.feat_weight * loss_feat
        
        loss = loss_cls + loss_kd + loss_feat
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_kd.update(loss_kd.item(), inputs.size(0))
        train_loss_feat.update(loss_feat.item(), inputs.size(0))
        
        top1, top5 = correct_num(logits, targets, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += targets.size(0)

        if args.rank == 0:
            print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, CLS Loss:{:.2f},' 
                'KD Loss:{:.2f}, Feature Loss:{:.2f}, Top-1 Acc:{:.2f}'.format(
                epoch, batch_idx, len(train_loader), lr, time.time()-batch_start_time, 
                train_loss_cls.avg, train_loss_kd.avg, train_loss_feat.avg, 
                (top1_num/total*100.).item()))
    
    acc1 = top1_num / total
    acc5 = top5_num / total

    if args.rank == 0:
        args.logger.info('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                    '\n Train_loss:{:.5f}'
                    '\t Train_loss_cls:{:.5f}'
                    '\t Train_loss_kd:{:.5f}'
                    '\t Train_loss_feat:{:.5f}'
                    '\nTrain top-1 accuracy:{:.2f}'
                    .format(epoch, lr, time.time() - start_time,
                            train_loss.avg,
                            train_loss_cls.avg,
                            train_loss_kd.avg,
                            train_loss_feat.avg,
                            acc1*100.))



def train(train_loader, model, criterion_list, optimizer, epoch, device, 
          args, agent, feat_trans, teacher_models, agent_optimizer):
    
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_kd = AverageMeter('train_loss_kd', ':.4e')
    train_loss_feat = AverageMeter('train_loss_feat', ':.4e')

    top1_num = 0
    top5_num = 0
    total = 0

    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_div = criterion_list[1]

    model.train()
    agent.eval()
    agent_states = []
    logits_agent_actions = []
    feature_agent_actions = []
    agent_rewards = []
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start_time = time.time()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        features, logits = model(inputs, is_feat=True) 
        trans_student_features = feat_trans(features[-2])
        
        teacher_logits = []
        teacher_features = []
        teacher_embeddings = []
        with torch.no_grad():
            all_teacher_info = []
            for t_model in teacher_models:
                t_features, t_logits = t_model(inputs, is_feat=True)
                t_feature = t_features[-1]
                t_feature = t_feature.detach() 
                t_logits = t_logits.detach() 
                
                teacher_features.append(t_features[-2])
                teacher_logits.append(t_logits)
                teacher_embeddings.append(t_features[-1])
                teacher_info = []
                teacher_info.append(t_feature)
                teacher_info.append(t_logits)
                teacher_info.append(F.cross_entropy(t_logits, targets, reduction='none').unsqueeze(-1)) # 128*1
                teacher_info = torch.cat(teacher_info, dim=1) # teacher logtis , teacher feature , CE_loss , student_teacher_gap
                all_teacher_info.append(teacher_info)

        agent_state = get_agent_state(trans_student_features, teacher_embeddings, logits, teacher_logits, targets, criterion_div)
        
        agent_states.append(agent_state) # [bx3, bx3, bx3]
        
        with torch.no_grad():
            logits_actions, feature_actions = agent(agent_state)
        if epoch == 0:
            logits_actions = torch.ones_like(logits_actions).cuda(args.gpu)
            feature_actions = torch.ones_like(feature_actions).cuda(args.gpu)
        logits_actions = logits_actions.detach() # batch_size x teacher_number
        feature_actions = feature_actions.detach()

        logits_agent_actions.append(logits_actions)
        feature_agent_actions.append(feature_actions)

        if args.rank == 0 and batch_idx % 10 == 0:
            #print('actions:{}'.format(str(actions)))
            args.logger.info('actions:{}'.format(str(logits_actions[0])))
            #args.logger.info('actions:{}'.format(str(logits_actions.max().item())+str(logits_actions.argmax(dim=1))))
        
        loss_cls = criterion_ce(logits, targets)
        
        loss_kd = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(teacher_models)):
            loss_kd = loss_kd + (logits_actions[:, idx] * criterion_div(logits, teacher_logits[idx].detach(), unreduce=True)).mean()
        loss_feat = torch.tensor(0.).cuda(args.gpu)
        
        if args.feat_kd == 'mse':
            feat_kd_func = FeatureMSELoss()
        elif args.feat_kd == 'kl':
            feat_kd_func = FeatureKLLoss(args.kd_T)

        for idx in range(len(teacher_models)):
            loss_feat = loss_feat +  (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx])).mean()

        loss_feat = args.feat_weight * loss_feat
        
        loss = loss_cls + loss_kd + loss_feat
        loss.backward()
        optimizer.step()

        
        sample_ce_loss = F.cross_entropy(logits, targets, reduction='none')
        sample_kd_loss = torch.tensor(0.).cuda(args.gpu)
        sample_feat_loss = torch.tensor(0.).cuda(args.gpu)
        for idx in range(len(teacher_models)):
            sample_kd_loss = sample_kd_loss + logits_actions[:, idx] * criterion_div(logits, teacher_logits[idx].detach(), unreduce=True)
            sample_feat_loss = sample_feat_loss + (feature_actions[:, idx] * feat_kd_func(trans_student_features[idx], teacher_features[idx]))
        reward = -(sample_ce_loss + sample_kd_loss+ args.feat_weight * sample_feat_loss)
        rewards_mean = reward.mean() 
        rewards_std = reward.std() 
        normalized_reward = (reward - rewards_mean) / rewards_std 
        normalized_reward = normalized_reward.detach()
        normalized_reward = torch.clamp(normalized_reward, min=0, max=1)
        #print('normalized_reward', normalized_reward)
        agent_rewards.append(normalized_reward)
        
        
        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))
        train_loss_kd.update(loss_kd.item(), inputs.size(0))
        train_loss_feat.update(loss_feat.item(), inputs.size(0))
        
        top1, top5 = correct_num(logits, targets, topk=(1, 5))
        top1_num += top1
        top5_num += top5
        total += targets.size(0)

        if args.rank == 0:
            print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}, CLS Loss:{:.2f},' 
                'KD Loss:{:.2f}, Feature Loss:{:.2f}, Top-1 Acc:{:.2f}'.format(
                epoch, batch_idx, len(train_loader), lr, time.time()-batch_start_time, 
                train_loss_cls.avg, train_loss_kd.avg, train_loss_feat.avg, 
                (top1_num/total*100.).item()))
        if batch_idx % args.agent_step == 0 and batch_idx != 0:
            train_agent(args, epoch, agent_states, agent_rewards, logits_agent_actions, agent, agent_optimizer)
            agent_states = []
            logits_agent_actions = []
            feature_agent_actions = []
            agent_rewards = []
        
    
    acc1 = top1_num / total
    acc5 = top5_num / total

    if args.rank == 0:
        args.logger.info('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                    '\n Train_loss:{:.5f}'
                    '\t Train_loss_cls:{:.5f}'
                    '\t Train_loss_kd:{:.5f}'
                    '\t Train_loss_feat:{:.5f}'
                    '\nTrain top-1 accuracy:{:.2f}'
                    .format(epoch, lr, time.time() - start_time,
                            train_loss.avg,
                            train_loss_cls.avg,
                            train_loss_kd.avg,
                            train_loss_feat.avg,
                            acc1*100.))
    if len(agent_states) != 0:
        train_agent(args, epoch, agent_states, agent_rewards, logits_agent_actions, agent, agent_optimizer)


def test(epoch, net, device, val_loader, criterion_ce, args, verbose=True):
    test_loss_cls = AverageMeter('Loss', ':.4e')
    top1_num = 0
    top5_num = 0
    total = 0
    
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            batch_start_time = time.time()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            features, logits = net(inputs, is_feat=True)
            loss_cls = torch.tensor(0.).cuda(args.gpu)
            loss_cls = criterion_ce(logits, targets)

            test_loss_cls.update(loss_cls.item(), inputs.size(0))

            top1, top5 = correct_num(logits, targets, topk=(1, 5))
            top1_num += top1
            top5_num += top5
            total += targets.size(0)
            
            if args.rank == 0 and verbose:
                print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Test Top-1 Acc:{:.4f}'.format(
                    epoch, batch_idx, len(val_loader), time.time()-batch_start_time, (top1_num/(total)*100.).item()))
            
        class_acc1 = round((top1_num/total*100.).item(), 4)
        class_acc5 = round((top5_num/total*100.).item(), 4)

        if args.rank == 0 and verbose:
            args.logger.info('Test epoch:{}\t Test_loss_cls:{:.5f}\nTest top-1 accuracy: {}\nTest top-5 accuracy: {}'
                        .format(epoch, test_loss_cls.avg, str(class_acc1), str(class_acc5)))
    return class_acc1
    