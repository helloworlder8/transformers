import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr

# pip install einops timm

import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# from transformers import Florence2ForConditionalGeneration
model = AutoModelForCausalLM.from_pretrained("hub_pull/microsoft/Florence-2-base", torch_dtype=torch_dtype).to(device)
processor = AutoProcessor.from_pretrained("hub_pull/microsoft/Florence-2-base")

prompt = "<OD>"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype) #torch.Size([1, 13]) torch.Size([1, 13]) torch.Size([1, 3, 768, 768])
from transformers import Florence2ForConditionalGeneration
generated_ids = model.generate( #torch.Size([1, 23])
    input_ids=inputs["input_ids"], #torch.Size([1, 13])
    pixel_values=inputs["pixel_values"], #torch.Size([1, 3, 768, 768])
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0] #'</s><s>car<loc_53><loc_333><loc_933><loc_774>door handle<loc_425><loc_503><loc_474><loc_515>wheel<loc_709><loc_576><loc_865><loc_772><loc_150><loc_584><loc_309><loc_773></s>'

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)
