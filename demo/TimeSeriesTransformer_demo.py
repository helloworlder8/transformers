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
from transformers import TimeSeriesTransformerModel

file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly")

outputs = model(
    past_values=batch["past_values"], #torch.Size([64, 61]) 过去的值 batch seq_len
    past_time_features=batch["past_time_features"], #torch.Size([64, 61, 2]) 过去的时间特征  每个时间步会有两个时间特征
    past_observed_mask=batch["past_observed_mask"], #torch.Size([64, 61]) 过去的观测掩码
    static_categorical_features=batch["static_categorical_features"], #torch.Size([64, 1]) 静态类别特征 地区、产品类型或客户分群等 理解成大的不同
    static_real_features=batch["static_real_features"], #torch.Size([64, 1]) 静态实值特征 影响时间序列的固定因素（如常量的气候变量或静态的经济指标等） 理解成小的不同
    future_values=batch["future_values"],#torch.Size([64, 24]) 未来的值
    future_time_features=batch["future_time_features"], #torch.Size([64, 24, 2]) 未来的时间特征
)

last_hidden_state = outputs.last_hidden_state
