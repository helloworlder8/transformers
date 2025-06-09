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

lcoal_image_path = "demo/sam_image.jpg"
# img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
raw_image = Image.open(lcoal_image_path).convert("RGB")
input_points = [[[400, 650]]]

# torch.Size([1, 2]) tensor([[1764, 2646]])
# torch.Size([1, 2, 4]) tensor([[[0.0620, 0.2252, 0.9527, 0.7112],
#          [0.0654, 0.2245, 0.9543, 0.7109]]])
# torch.Size([1, 2, 4]) tensor([[[ 109.4105,  595.7709, 1680.6196, 1881.8335],
        #  [ 115.2851,  593.9832, 1683.3145, 1880.9552]]])
input_boxes = [[[ 109.4105,  595.7709, 1680.6196, 1881.8335],[ 115.2851,  593.9832, 1683.3145, 1880.9552]]]
inputs = processor(images=raw_image, input_points=None, input_boxes= input_boxes, return_tensors="pt")
from transformers import SamModel
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(
    outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
)
