import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr


from PIL import Image
import requests
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("facebook/sam-vit-base")
processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

lcoal_image_path = "image.png"
# img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
raw_image = Image.open(lcoal_image_path).convert("RGB")
input_points = [[[400, 650]]]
inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")
from transformers import SamModel
outputs = model(**inputs)

masks = processor.post_process_masks(
    outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
)
