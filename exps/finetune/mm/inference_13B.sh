#!/bin/bash
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1

PRETRAINED=/home/cx/llama2_accessory/LLaMA2-Accessory-main/output/common/llama2_qformer_13B_aokvqa/epoch2
PRETRAINED_BASE=/home/cx/ckpts/llama2_acc/alpacaLlava_llamaQformerv2Peft_13b
LLAMA_CONFIG="/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/13B_params.json"
TOKENIZER=/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/tokenizer.model


data_parallel=fsdp
model_parallel=1

CUDA_VISIBLE_DEVICES=5 torchrun --nproc-per-node=1 --master-port=11120 /home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/demos/single_turn_mm_cloud.py \
--pretrained_path $PRETRAINED --pretrained_path_base $PRETRAINED_BASE --llama_type llama_qformerv2_cloud --llama_config $LLAMA_CONFIG --tokenizer_path $TOKENIZER
