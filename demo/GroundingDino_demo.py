import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr

from transformers import AutoProcessor
model_id = "IDEA-Research/grounding-dino-tiny" #The International Digital Economy Academy
from transformers import GroundingDinoProcessor
processor = AutoProcessor.from_pretrained(model_id)


import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


text = "a Cockroach." #一句话给出最大概率的值
image_path = "/18t/data/home/ang/ang/paper4/EVF-SAM/Ang/EASam/Cockroach_1.png"
image = Image.open(image_path).convert("RGB")
device = "cuda:1"

# huggingface-cli download --resume-download IDEA-Research/grounding-dino-tiny --local-dir ./ --local-dir-use-symlinks False

inputs = processor(images=image, text=text, return_tensors="pt").to(device) 

model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

from transformers import GroundingDinoForObjectDetection
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