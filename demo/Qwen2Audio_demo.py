from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor,PreTrainedModel
# from transformers import Qwen2AudioProcessor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct-Int4")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct-Int4", device_map="auto")

conversation = [ #包含角色和内容（包含类别和实际内容）
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"},
    ]},
    {"role": "assistant", "content": "Yes, the speaker is female and in her twenties."},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav"},
    ]},
]

""" 
processor 作用 
1 从字典处理生成文本和音频
2 从文本和音频处理成输入数据
3 从输出输入处理成实际输出
"""
# <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
# <|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n<|im_end|>\n
# <|im_start|>assistant\nYes, the speaker is female and in her twenties.<|im_end|>\n
# <|im_start|>user\nAudio 2: <|audio_bos|><|AUDIO|><|audio_eos|>\n<|im_end|>\n
# <|im_start|>assistant\n
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) # 

# [(144000,)(203520,)]  
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(librosa.load(
                    BytesIO(urlopen(ele['audio_url']).read()), 
                    sr=processor.feature_extractor.sampling_rate)[0]
                )

#torch.Size([1, 58]) torch.Size([1, 58]) torch.Size([2, 128, 3000]) torch.Size([2, 3000])
inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
inputs.input_ids = inputs.input_ids.to("cuda")

generate_ids = model.generate(**inputs, max_length=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]



