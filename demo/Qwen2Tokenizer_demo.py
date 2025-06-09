from transformers import Qwen2Tokenizer

tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-tokenizer")
inpute = tokenizer("Hello world")["input_ids"]
# [9707, 1879]
print(inpute)
inpute = tokenizer(" Hello world")["input_ids"]
# [21927, 1879]
print(inpute)