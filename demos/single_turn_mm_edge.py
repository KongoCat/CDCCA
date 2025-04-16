import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

from model.meta_edge import MetaModel
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
from pt_transporter import PTSender

def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama_qformerv2', type=str, metavar='MODEL',
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
target_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}[args.dtype]

with default_tensor_type(dtype=target_dtype, device="cpu" if args.quant else "cuda"):
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)
#exit()
print(f"load pretrained from {args.pretrained_path}")
load_result = load_tensor_parallel_model_list(model, args.pretrained_path_base)
# print("load result: ", load_result)
load_result = load_tensor_parallel_model_list(model, args.pretrained_path)
# print("load result: ", load_result)


if args.quant:
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
model.bfloat16().cuda() #16bit debug

@ torch.inference_mode()
def generate(
        img_path,
        prompt,
        question_input,
        system_prompt,
        max_gen_len,
        gen_t, top_p
):
   
    if img_path is not None:
        image = Image.open(img_path).convert('RGB')
        image = transform_val(image).unsqueeze(0)
    else:
        image = None

    # text output
    _prompt = format_prompt({"instruction":prompt, "input":question_input}, system_prompt)

    dist.barrier()
    dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])

    if image is not None:
        image = image.cuda()
    with torch.cuda.amp.autocast(dtype=target_dtype):
        results, probs, image_tokens = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
    text_output = results[0].strip()
    return text_output, probs, image_tokens


failed_samples = [] 

#可以替换为一系列数据循环，单帧图像用于调试
with open("/home/cx/datasets/cdv3/tpami_json/common/okvqa_train.json", "r") as f:
    data = json.load(f)
    
l = len(data)
print(l)
for i in tqdm(data[:l//20]):
    img_path = i["image"] #'/data/lcx/someResource/man.jpeg'
    question = i["conversations"][0]["value"] #'What is in the photo?'
    print(img_path)
    ans, probs, image_tokens = generate(img_path, question, "","alpaca",128,0.1,.75)
    # print(ans, probs) #debug
    conf_stack = torch.stack(probs)
    conf = torch.prod(conf_stack).item()

    #UTS-1 & UTS-2
    if conf < 0.5:
        print("something!")
        # 可定义在外部或提前初始化
        sample = {
            "img_path": img_path,
            "question": question,
            "token_tensor": image_tokens,
            "prob": conf
        }
        failed_samples.append(sample)

#torch.save(failed_samples,"edge_UTS.pt") # local debug
sender = PTSender(server_ip='127.0.0.1', port=7007) #localhost debug
for i, sample in enumerate(failed_samples):
    tensor = sample["token_tensor"]
    print(f"[Sender] Sample {i}: token_tensor shape = {tensor.shape}")

sender.send(failed_samples)
