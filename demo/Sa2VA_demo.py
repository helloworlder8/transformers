import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr




import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
import os
import requests

# from transformers import Sa2VACha`tModel
# load the model and tokenizer
path = "hub_pull/ByteDance/Sa2VA-1B"
device = "cuda"
from transformers import Qwen2Tokenizer
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(
    path, #'ByteDance/Sa2VA-1B'
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=None).eval().to(device)


# # for image chat
# image_path = "/PATH/TO/IMAGE"
# text_prompts = "<image>Please describe the image."
# image = Image.open(image_path).convert('RGB')
# input_dict = {
#     'image': image,
#     'text': text_prompts,
#     'past_text': '',
#     'mask_prompts': None,
#     'tokenizer': tokenizer,
#     }
# return_dict = model.predict_forward(**input_dict)
# answer = return_dict["prediction"] # the text format answer

image_path = "demo/image.jpg"
image = Image.open(image_path).convert('RGB')
# for image chat with segmentation output
text_prompts = "<image>please segment the cat."
input_dict = {
    'image': image,
    'text': text_prompts,
    'past_text': '',
    'mask_prompts': None,
    'tokenizer': tokenizer,
    }
from transformers import Sa2VAChatModel
return_dict = model.predict_forward(**input_dict)
answer = return_dict["prediction"] # the text format answer
masks = return_dict['prediction_masks']  # segmentation masks, list(np.array(1, h, w), ...)
    
# for chat with visual prompt (mask format) input
mask_prompts = np.load('/PATH/TO/pred_masks.npy') # np.array(n_prompts, h, w)
image_path = "/PATH/TO/IMAGE"
text_prompts = "<image>Can you provide me with a detailed description of the region in the picture marked by region1."
image = Image.open(image_path).convert('RGB')
input_dict = {
    'image': image,
    'text': text_prompts,
    'past_text': '',
    'mask_prompts': mask_prompts,
    'tokenizer': tokenizer,
    }
return_dict = model.predict_forward(**input_dict)
answer = return_dict["prediction"] # the text format answer

# for video chat
video_folder = "/PATH/TO/VIDEO_FOLDER"
images_paths = os.listdir(video_folder)
images_paths = [os.path.join(video_folder, image_path) for image_name in images_paths]
if len(images_paths) > 5:  # uniformly sample 5 frames
    step = (len(images_paths) - 1) // (5 - 1)
    images_paths = [images_paths[0]] + images_paths[1:-1][::step][1:] + [images_paths[-1]]
text_prompts = "<image>Please describe the video."
input_dict = {
    'video': images_paths,
    'text': text_prompts,
    'past_text': '',
    'mask_prompts': None,
    'tokenizer': tokenizer,
}
return_dict = model.predict_forward(**input_dict)
answer = return_dict["prediction"] # the text format answer


# for video chat with segmentation mask output
video_folder = "/PATH/TO/VIDEO_FOLDER"
images_paths = os.listdir(video_folder)
images_paths = [os.path.join(video_folder, image_path) for image_name in images_paths]
text_prompts = "<image>Please segment the person."
input_dict = {
    'video': images_paths,
    'text': text_prompts,
    'past_text': '',
    'mask_prompts': None,
    'tokenizer': tokenizer,
}
return_dict = model.predict_forward(**input_dict)
answer = return_dict["prediction"] # the text format answer
masks = return_dict['prediction_masks']  # segmentation masks, list(np.array(n_frames, h, w), ...)
