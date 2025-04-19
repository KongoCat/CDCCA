import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

from model.meta_cloud import MetaModel
import pandas as pd
import argparse
import torch
import torch.distributed as dist
import gradio as gr

from PIL import Image

from util import misc
from fairscale.nn.model_parallel import initialize as fs_init

from data.alpaca import transform_val, format_prompt
from util.tensor_parallel import load_tensor_parallel_model_list
from util.tensor_type import default_tensor_type
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import json
from pt_transporter import PTReceiver
import yaml



def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama_qformerv2_cloud', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')

    parser.add_argument('--pretrained_path_base', default='/path/to/pretrained', type=str, nargs="+",
                        help='directory containing pre-trained checkpoints')
    
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                        help='directory containing pre-trained checkpoints')

    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument('--model_parallel_size', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="fp16",
                        help="The dtype used for model weights and inference.")
    parser.add_argument('--quant', action='store_true', help="enable quantization")
    return parser




args = get_args_parser().parse_args()

# define the model
misc.init_distributed_mode(args)
fs_init.initialize_model_parallel(args.model_parallel_size)

receiver = PTReceiver(listen_ip='0.0.0.0', port=7010)
if dist.get_rank() == 0:
    data = receiver.receive()
    obj_list = [data]
    dist.broadcast_object_list(obj_list, src=0)
else:
    print(f"[Rank {dist.get_rank()}] Waiting for rank 0...")
    obj_list = [None]
    dist.broadcast_object_list(obj_list, src=0)
    data = obj_list[0]
    print(f"[Rank {dist.get_rank()}] Received data from rank 0!")


target_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}[args.dtype]

with default_tensor_type(dtype=target_dtype, device="cpu" if args.quant else "cuda"):
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)
#exit()
print(f"load pretrained from {args.pretrained_path}")
load_result = load_tensor_parallel_model_list(model, args.pretrained_path_base + args.pretrained_path)
# print("load result: ", load_result)
# load_result = load_tensor_parallel_model_list(model, args.pretrained_path)
# print("load result: ", load_result)


if args.quant: #DWC
    print("Quantizing model to 4bit!")
    from util.quant import quantize
    from transformers.utils.quantization_config import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig.from_dict(
        config_dict={
            "load_in_8bit": False,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        },
        return_unused_kwargs=False,
    )
    quantize(model, quantization_config)

print("Model = %s" % str(model))
model.bfloat16().cuda() #debug 16bit

@ torch.inference_mode()
def generate(
        token_tensor,
        prompt,
        question_input,
        system_prompt,
        max_gen_len,
        gen_t, top_p
):
   
    # text output
    _prompt = format_prompt({"instruction":prompt, "input":question_input}, system_prompt)

    dist.barrier()
    dist.broadcast_object_list([_prompt, token_tensor, max_gen_len, gen_t, top_p]) #token_tensor must load on cpu

    if token_tensor is not None:
        toekn_tensor = token_tensor.cuda() # UTS tokens
        
    with torch.cuda.amp.autocast(dtype=target_dtype):
        results = model.generate([_prompt], token_tensor, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
        
    text_output = results[0].strip()
    return text_output


structured_samples = []
save_dir = "/home/cx/llama2_accessory/someCode/debug"
os.makedirs(save_dir, exist_ok=True)


for i,sample in enumerate(data):
    question = sample["question"]
    token_tensor = sample["token_tensor"]
    local_rank = torch.cuda.current_device()
    token_tensor = token_tensor.to(f"cuda:{local_rank}")
    ans = generate(token_tensor, question, "","alpaca",128,0.1,.75)
    # 保存 token_tensor 为单独的 .pt 文件
    tensor_path = os.path.join(save_dir, f"token_tensor_{i:05d}.pt")
    torch.save(token_tensor, tensor_path)
    
    entry = {
        "id": f"sample_{i:05d}",
        "image": tensor_path,  # 替换原本的图像路径
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": ans}
        ]
    }
    
    structured_samples.append(entry)

# 将所有结构写入 JSON 文件
output_json_path = "/home/cx/datasets/cdv3/tpami_json/common/debug.json"
with open(output_json_path, "w") as f:  #pseudo label
    json.dump(structured_samples, f, indent=2)
    
# 读取原始 YAML 文件
with open("/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/data/finetune/mm/alpaca_llava_cdcca.yaml", "r") as f:
    config = yaml.safe_load(f)

# 修改 path 字段
config["META"][0]["path"] = output_json_path

# 保存回 YAML 文件（覆盖或另存）
with open("/home/cx/llama2_accessory/LLaMA2-Accessory-main/accessory/configs/data/finetune/mm/alpaca_llava_cdcca.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False)

print(f"Updated META[0].path to: {output_json_path}")
