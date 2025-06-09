import os

os.environ["TRANSFORMERS_CACHE"] = "/media/ang/2T/hf_models"
os.environ["HF_DATASETS_CACHE"] = "/media/ang/2T/hf_datasets"

import torch
original_repr = torch.Tensor.__repr__ #保存原始函数函数
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr



from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name) #class Qwen2TokenizerFast
model = AutoModelForCausalLM.from_pretrained(
    model_name, #'Qwen/Qwen3-0.6B'
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model." #str
messages = [
    {"role": "user", "content": prompt} #list dict
]
text = tokenizer.apply_chat_template(
    messages, #【{}】
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device) #->{'input_ids' 'attention_mask'}

# conduct text completion
generated_ids = model.generate( #->torch.Size([257])包含输入信息
    **model_inputs, #{'input_ids': torch.Size([1, 18]), 'attention_mask': torch.Size([1, 18])}
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() #[len221]

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
