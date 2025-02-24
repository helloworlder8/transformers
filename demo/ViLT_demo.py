import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr


""" ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision """
""" 第一个摆脱了目标检测的视觉文本模型 """
from transformers import ViltProcessor, ViltModel
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "hello world"

model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")


inputs = processor(image, text, return_tensors="pt")
from transformers import ViltModel
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
