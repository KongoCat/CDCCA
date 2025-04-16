#!/bin/bash

pretrained_path=$1
pretrained_type=consolidated
llama_config="$2"
tokenizer_path="$3"
data_config=/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/data/finetune/mm/alpaca_llava_cdcca.yaml

data_parallel=fsdp
model_parallel=1

exp_name=cdccav2/debug
echo "exp name: $exp_name"
mkdir -p /home/cx/ckpts/TPAMI/"$exp_name"

CUDA_VISIBLE_DEVICES=5 torchrun --master_port=1112 --nproc_per_node=1 /home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/main_finetune_CDCCA.py \
--output_dir /home/cx/ckpts/TPAMI/"$exp_name" --epochs 3 --warmup_epochs 0.2 \
--batch_size 1 --accum_iter 1 --num_workers 4 \
--max_words 512 \
--lr 0.00003 --min_lr 0.000005 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel"  --checkpointing \
--llama_type llama_qformerv2_cloud --llama_config "/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/7B_params.json" --tokenizer_path "/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/tokenizer.model" \
--pretrained_path "/home/cx/ckpts/llama2_acc/alpacaLlava_llamaQformerv2" --pretrained_type="$pretrained_type" \
--data_config $data_config \
--only_save_trainable \
2>&1 | tee -a /home/cx/ckpts/TPAMI/"$exp_name"/output.log

echo "exp name: $exp_name"