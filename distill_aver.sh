python train_student_avg.py \
    --data "./data" \
    --arch ShuffleV2 \
    --checkpoint-dir "./ckpt" \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
    --dist-backend 'nccl' \
    --world-size 1 \
    --rank 0 

python train_student_avg.py \
    --data "./data" \
    --arch RegNetX_200MF \
    --checkpoint-dir "./ckpt" \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
    --dist-backend 'nccl' \
    --world-size 1 \
    --rank 0 