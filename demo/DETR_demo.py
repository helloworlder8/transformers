import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr


from transformers import AutoImageProcessor, DetrModel
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = DetrModel.from_pretrained("facebook/detr-resnet-50")
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")


inputs = image_processor(images=image, return_tensors="pt") #torch.Size([1, 3, 800, 1066]) torch.Size([1, 800, 1066])


from transformers import DetrModel
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
