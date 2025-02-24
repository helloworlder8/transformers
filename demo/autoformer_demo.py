import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr



from huggingface_hub import hf_hub_download
import torch
from transformers import AutoformerModel

file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file) #字典

model = AutoformerModel.from_pretrained("huggingface/autoformer-tourism-monthly")

from transformers import AutoformerModel
outputs = model(
    past_values=batch["past_values"], #torch.Size([64, 61])
    past_time_features=batch["past_time_features"], #torch.Size([64, 24, 2])
    past_observed_mask=batch["past_observed_mask"], #torch.Size([64, 24])
    static_categorical_features=batch["static_categorical_features"], #torch.Size([64, 1])
    future_values=batch["future_values"],#torch.Size([64, 24])
    future_time_features=batch["future_time_features"], #torch.Size([64, 24, 2])
)

last_hidden_state = outputs.last_hidden_state
