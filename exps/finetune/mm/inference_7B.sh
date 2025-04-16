#!/bin/bash
PRETRAINED=/home/cx/ckpts/TPAMI/common/llama2_qformer_7B_aokvqa/epoch0
PRETRAINED_BASE=/home/cx/ckpts/llama2_acc/alpacaLlava_llamaQformerv2
LLAMA_CONFIG="/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/7B_params.json"
TOKENIZER=/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/tokenizer.model


data_parallel=fsdp
model_parallel=1

CUDA_VISIBLE_DEVICES=4 torchrun --nproc-per-node=1 --master-port=11112 /home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/demos/single_turn_mm_edge.py \
--pretrained_path $PRETRAINED --pretrained_path_base $PRETRAINED_BASE --llama_type llama_qformerv2_edge --llama_config $LLAMA_CONFIG --tokenizer_path $TOKENIZER
