# # Tokenizer 基本使用
from transformers import AutoTokenizer
sen = "弱小的我也有大梦想!"
# ## Step1 加载与保存
# 从HuggingFace加载，输入模型名称，即可加载对于的分词器
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
tokenizer
# tokenizer 保存到本地
tokenizer.save_pretrained("./roberta_tokenizer")
# 从本地加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./roberta_tokenizer/")
tokenizer
# ## Step2 句子分词
tokens = tokenizer.tokenize(sen)
tokens
# ## Step3 查看词典
tokenizer.vocab
tokenizer.vocab_size
# ## Step4 索引转换
# 将词序列转换为id序列
ids = tokenizer.convert_tokens_to_ids(tokens)
ids
# 将id序列转换为token序列
tokens = tokenizer.convert_ids_to_tokens(ids)
tokens
# 将token序列转换为string
str_sen = tokenizer.convert_tokens_to_string(tokens)
str_sen
# ###  更便捷的实现方式
# 将字符串转换为id序列，又称之为编码
ids = tokenizer.encode(sen, add_special_tokens=True)
ids
# 将id序列转换为字符串，又称之为解码
str_sen = tokenizer.decode(ids, skip_special_tokens=False)
str_sen
# ## Step5 填充与截断
# 填充
ids = tokenizer.encode(sen, padding="max_length", max_length=15)
ids
# 截断
ids = tokenizer.encode(sen, max_length=5, truncation=True)
ids
# ## Step6 其他输入部分
ids = tokenizer.encode(sen, padding="max_length", max_length=15)
ids
attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids = [0] * len(ids)
ids, attention_mask, token_type_ids
# ## Step7 快速调用方式
inputs = tokenizer.encode_plus(sen, padding="max_length", max_length=15)
inputs
inputs = tokenizer(sen, padding="max_length", max_length=15)
inputs
# ## Step8 处理batch数据
sens = ["弱小的我也有大梦想",
        "有梦想谁都了不起",
        "追逐梦想的心，比梦想本身，更可贵"]
res = tokenizer(sens)
res

# 单条循环处理
for i in range(1000):
    tokenizer(sen)

# 处理batch数据
res = tokenizer([sen] * 1000)
tokenizer
# # Fast / Slow Tokenizer
sen = "弱小的我也有大Dreaming!"
fast_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
fast_tokenizer
slow_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese", use_fast=False)
slow_tokenizer

# 单条循环处理
for i in range(10000):
    fast_tokenizer(sen)

# 单条循环处理
for i in range(10000):
    slow_tokenizer(sen)

# 处理batch数据
res = fast_tokenizer([sen] * 10000)

# 处理batch数据
res = slow_tokenizer([sen] * 10000)
inputs = fast_tokenizer(sen, return_offsets_mapping=True)
inputs
inputs.word_ids()
inputs = slow_tokenizer(sen, return_offsets_mapping=True)
# # 特殊Tokenizer的加载
from transformers import AutoTokenizer
# 新版本的transformers（>4.34），加载 THUDM/chatglm 会报错，因此这里替换为了天宫的模型
tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-13B-base", trust_remote_code=True)
tokenizer
tokenizer.save_pretrained("skywork_tokenizer")
tokenizer = AutoTokenizer.from_pretrained("skywork_tokenizer", trust_remote_code=True)
tokenizer.decode(tokenizer.encode(sen))

