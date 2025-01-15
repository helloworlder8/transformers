""" Qwen2Audio重点 """


""" 千文音频大模型处理器类 """
# Qwen2TokenizerFast Tokenizer处理器   WhisperFeatureExtractor音频特征提取器
from transformers import Qwen2AudioProcessor,Qwen2TokenizerFast,WhisperFeatureExtractor


from transformers import Qwen2AudioForConditionalGeneration, GenerationMixin, GenerationConfig


""" 模型前向传播 """
            # # forward pass to get next token
            # outputs = self(**model_inputs, return_dict=True)

""" 通义千文配置 """
from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioConfig, Qwen2AudioEncoderConfig, Qwen2Config

# Initializing a Qwen2AudioEncoder config
audio_config = Qwen2AudioEncoderConfig() #音频配置

# Initializing a Qwen2 config
text_config = Qwen2Config() #文本配置

# Initializing a Qwen2Audio configuration
configuration = Qwen2AudioConfig(audio_config, text_config) #整体配置

# Initializing a model from the qwen2-audio style configuration
model = Qwen2AudioForConditionalGeneration(configuration) #生成模型

# Accessing the model configuration
configuration = model.config #传回模型配置







""" 全局重点 """

""" #加载预训练的模型 """
from transformers import PreTrainedModel
PreTrainedModel._load_pretrained_model
