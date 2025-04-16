#!/bin/bash

pretrained_path=$1
pretrained_type=consolidated
llama_config="$2 configs/model/finetune/sg/llamaPeft_normBiasLora.json"
tokenizer_path="$3"
data_config=/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/data/finetune/mm/alpaca_llava_rsvqa.yaml

data_parallel=sdp
model_parallel=2

exp_name=remotesensing/llama2_qformer_13B_rsvqa
echo "exp name: $exp_name"
mkdir -p /home/cx/ckpts/TPAMI/"$exp_name"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=11112 --nproc_per_node=4 /home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/main_finetune.py \
--output_dir /home/cx/ckpts/TPAMI/"$exp_name" --epochs 3 --warmup_epochs 0.2 \
--batch_size 1 --accum_iter 2 --num_workers 4 \
--max_words 512 \
--lr 0.00005 --min_lr 0.000005 --clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" --checkpointing \
--llama_type llama_qformerv2_peft --llama_config "/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/13B_params.json" --tokenizer_path "/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/tokenizer_13B.model" \
--pretrained_path "/home/cx/ckpts/llama2_acc/alpacaLlava_llamaQformerv2Peft_13b" --pretrained_type="$pretrained_type" \
--data_config $data_config \
--only_save_trainable \
# --resume "/home/cx/ckpts/TPAMI/remotesensing/llama2_qformer_13B_rsvqa/epoch0-iter9999" \
2>&1 | tee -a /home/cx/ckpts/TPAMI/"$exp_name"/output.log

echo "exp name: $exp_name"