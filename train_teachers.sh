python ./train_baseline.py --model RegNetY_400MF \
    --data-folder "./data" \
    --checkpoint-dir "./ckpt"

python ./train_baseline.py --model RegNetX_400MF \
    --data-folder "./data" \
    --checkpoint-dir "./ckpt"

python ./train_baseline.py --model resnet32x4 \
    --data-folder "./data" \
    --checkpoint-dir "./ckpt"

python ./train_baseline.py --model wrn_28_4 \
    --data-folder "./data" \
    --checkpoint-dir "./ckpt"