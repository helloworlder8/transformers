import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr

import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny" #The International Digital Economy Academy
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
text = "a cat. a remote control."
device = "cuda"


from transformers import GroundingDinoProcessor
processor = AutoProcessor.from_pretrained(model_id)

inputs = processor(images=image, text=text, return_tensors="pt").to(device) 

from transformers import GroundingDinoForObjectDetection
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
with torch.no_grad():
    outputs = model(**inputs) 


results = processor.post_process(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

processor.annotate(image, results)

print(results)