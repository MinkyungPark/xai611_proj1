## XAI611 Project1
2022020876 박민경

### Pretraining
- Graphormer model based on fairseq 0.12.2
- original implementation : https://github.com/microsoft/Graphormer
- Dataset : pcqm4mv2
```
CUDA_VISIBLE_DEVICES=3 fairseq-train \
    --user-dir /workspace/fairseq/examples/graphormer \
    --num-workers 16 \
    --ddp-backend=legacy_ddp \
    --dataset-name pcqm4mv2 \
    --dataset-source ogb \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_base \
    --num-classes 1 \
    --attention-dropout 0.1 \
    --act-dropout 0.1 \
    --dropout 0.0 \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-8 \
    --clip-norm 5.0 \
    --weight-decay 0.0 \
    --lr-scheduler polynomial_decay \
    --power 1 \
    --warmup-updates 60000 \
    --total-num-update 1000000 \
    --lr 2e-4 \
    --end-learning-rate 1e-9 \
    --batch-size 256 \
    --fp16 \
    --data-buffer-size 20 \
    --save-dir ./ckpts \
    --tensorboard-logdir /workspace/ckpts/tensorboard
```

### Finetuning
- Dataset : BBBP
```
CUDA_VISIBLE_DEVICES=3 fairseq-train \
    --user-dir /workspace/fairseq/examples/graphormer \
    --num-workers 16 \
    --ddp-backend=legacy_ddp \
    --dataset-name ogbg-molbbbp \
    --dataset-source ogb \
    --task graph_prediction_with_flag \
    --criterion binary_logloss_with_flag \
    --reset-optimizer \
    --arch graphormer_base \
    --num-classes 1 \
    --attention-dropout 0.1 \
    --act-dropout 0.1 \
    --dropout 0.0 \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-8 \
    --clip-norm 5.0 \
    --weight-decay 0.0 \
    --lr-scheduler polynomial_decay \
    --power 1 \
    --warmup-updates 6000 \
    --total-num-update 100000 \
    --lr 2e-4 \
    --end-learning-rate 1e-9 \
    --batch-size 256 \
    --fp16 \
    --data-buffer-size 20 \
    --save-dir ./ckpts_ft_craft \
    --pretrained-model ./ckpts \
    --tensorboard-logdir /workspace/ckpts_ft_craft/tensorboard
```

### Finetuning with huggingface prerained model
- Dataset : BBBP
```
CUDA_VISIBLE_DEVICES=3 fairseq-train \
    --user-dir /workspace/fairseq/examples/graphormer \
    --num-workers 16 \
    --ddp-backend=legacy_ddp \
    --dataset-name ogbg-molbbbp \
    --dataset-source ogb \
    --task graph_prediction_with_flag \
    --criterion binary_logloss_with_flag \
    --reset-optimizer \
    --arch graphormer_base \
    --num-classes 1 \
    --attention-dropout 0.1 \
    --act-dropout 0.1 \
    --dropout 0.0 \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-8 \
    --clip-norm 5.0 \
    --weight-decay 0.0 \
    --lr-scheduler polynomial_decay \
    --power 1 \
    --warmup-updates 6000 \
    --total-num-update 100000 \
    --lr 2e-4 \
    --end-learning-rate 1e-9 \
    --batch-size 256 \
    --fp16 \
    --data-buffer-size 20 \
    --encoder-layers 12 \
    --encoder-embed-dim 768 \
    --encoder-ffn-embed-dim 768 \
    --encoder-attention-heads 32 \
    --flag-m 3 \
    --flag-step-size 0.01 \
    --flag-mag 0 \
    --save-dir ./ckpts_ft \
    --pretrained-model-name pcqm4mv2_graphormer_base \
    --tensorboard-logdir /workspace/ckpts_ft/tensorboard
```


best_valid_loss_model [download_link](https://holy-danthus-8ed.notion.site/model-5f967dc35f364a8c8bfd7efef4afc72d)

### Evaluation
```
python examples/graphormer/evaluate/evaluate.py
    --user-dir /workspace/fairseq/examples/graphormer \
    --task graph_prediction \
    --arch graphormer_base \
    --num-classes 1 \
    --dataset-name ogbg-molbbbp \
    --dataset-source ogb \
    --split test \
    --metric auc \
    --save-dir /workspace/ckpts_ft/checkpoint_last.pt 
    \\ --pretrained-model-name pcqm4mv2_graphormer_base \ # for pretrained model from hf
    \\ --use-pretrained
```

#### Extra dependencies
```
torch                    1.12.0+cu113
torch-cluster            1.6.1
torch-geometric          2.3.1
torch-scatter            2.1.1
torch-sparse             0.6.17
torch-spline-conv        1.2.2
pyg-lib                  0.2.0+pt112cu113
lmdb                     1.4.1
ogb                      1.3.6
rdkit-pypi               2022.9.5
dgl                      1.0.1
```