import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr

from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests
from transformers import CLIPSegProcessor
processor = AutoProcessor.from_pretrained("hub_pull/CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("hub_pull/CIDAS/clipseg-rd64-refined")
# CIDAS/clipseg-rd64-refined
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a cat", "a remote", "a blanket"]
inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")
#有输入ids 注意力掩膜 像素值 CLIPSegForImageSegmentation
from transformers import CLIPSegForImageSegmentation
outputs = model(**inputs)

logits = outputs.logits
print(logits.shape)


# huggingface-cli download --resume-download CIDAS/clipseg-rd64-refined --local-dir CIDAS/clipseg-rd64-refined --local-dir-use-symlinks False