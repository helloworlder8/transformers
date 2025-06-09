import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr


""" BEIT: BERT Pre-Training of Image Transformers """
""" 对应bert 图像打成patch随机mask预测token """
from transformers import AutoImageProcessor, BeitForMaskedImageModeling
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

num_patches = (model.config.image_size // model.config.patch_size) ** 2 # (224/16)**2 196
pixel_values = image_processor(images=image, return_tensors="pt").pixel_values #torch.Size([1, 3, 224, 224]) 相当于只取其成员属性
bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool() #图像块(patch)随机mask

from transformers import BeitForMaskedImageModeling
outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
loss, logits = outputs.loss, outputs.logits
list(logits.shape)
