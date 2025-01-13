import torch.nn as nn

__all__ = ['FeatureKLLoss', 'FeatureMSELoss']


class FeatureMSELoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(FeatureMSELoss, self).__init__()

    def forward(self, f_s, f_t):
        f_s = f_s.view(f_s.size(0), -1)
        f_t = f_t.view(f_t.size(0), -1) 
        loss = ((f_s - f_t)**2).mean(1)
        return loss


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap
    
    
class FeatureKLLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super(FeatureKLLoss, self).__init__()
        self.normalize = ChannelNorm()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.temperature = temperature
       
    def forward(self, f_s, f_t):
        #n,c,h,w = f_s.shape
        norm_s = self.normalize(f_s/self.temperature)
        norm_t = self.normalize(f_t.detach()/self.temperature)
        norm_s = norm_s.log()

        loss = self.criterion(norm_s, norm_t).sum(-1).mean(-1)

        return loss * (self.temperature**2)