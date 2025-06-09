""" 资料 """
# https://huggingface.co/openai/clip-vit-large-patch14
import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr


""" demo1 """
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
#{'input_ids':torch.Size([2, 7]),  'attention_mask':torch.Size([2, 7]), 'pixel_values':torch.Size([1, 3, 224, 224])}
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities




""" demo2 """
# from PIL import Image
# import requests
# from transformers import AutoProcessor, CLIPVisionModel

# model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(images=image, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# pooled_output = outputs.pooler_output




""" demo3 """
# from transformers import AutoTokenizer, CLIPTextModel

# model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state #torch.Size([2, 7, 512])
# pooled_output = outputs.pooler_output #torch.Size([2, 512])






""" demo4 """
# from PIL import Image
# import requests
# from transformers import AutoProcessor
# from transformers.models.clip.modeling_clip import CLIPForImageClassification

# # 1. 加载图像分类模型（注意：你需要确保模型 checkpoint 是 image classification 类型）
# model = CLIPForImageClassification.from_pretrained("openai/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # 2. 加载图像
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# # 3. 图像预处理
# inputs = processor(images=image, return_tensors="pt")

# # 4. 推理
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class = logits.argmax(-1).item()

# # 5. 输出结果
# print("logits:", logits)
# print("Predicted class index:", predicted_class)
