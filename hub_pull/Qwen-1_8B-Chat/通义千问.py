from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_dir = '../Qwen-1_8B-Chat' #模型



""" 加载分词器 """
tokenizer = AutoTokenizer.from_pretrained(model_dir, revision='master', trust_remote_code=True) #本地加载分词器


""" 加载模型 """
# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat", device_map="cpu", trust_remote_code=True).eval()
model = AutoModelForCausalLM.from_pretrained(model_dir, revision='master', device_map="auto", trust_remote_code=True).eval() #本地加载模型




def formatted_response(response):
    # first_word = response[0]  # 按空格分割，取第一个词
    blue_bold = "\033[1;34m"  # 蓝色加粗
    reset = "\033[0m"         # 重置样式
    answer="answer"
    # formatted_response = f"{blue_bold}{first_word}{reset}{response[1:]}\n"
    formatted_response = f"{blue_bold}{answer}{reset}{response}\n"
    return formatted_response


""" 简单对话 """
response, history = model.chat(tokenizer, "你好", history=None) #->str list
response = formatted_response(response)
print(response) #并不是固定的



""" 输入加历史对话 """
# 第二轮对话 2nd dialogue turn
response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
response = formatted_response(response)
print(response)

response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
response = formatted_response(response)
print(response) #history是累计历史



""" System Prompt """
# Qwen-1.8B-Chat can realize roly playing, language style transfer, task setting, and behavior setting by system prompt.
response, _ = model.chat(tokenizer, "你好呀", history=None, system="请用二次元可爱语气和我说话")
response = formatted_response(response)
print(response)

response, _ = model.chat(tokenizer, "My colleague works diligently", history=None, system="You will write beautiful compliments according to needs")
response = formatted_response(response)
print(response)
