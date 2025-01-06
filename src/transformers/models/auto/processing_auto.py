# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AutoProcessor class."""

import importlib
import inspect
import json
import os
import warnings
from collections import OrderedDict

# Build the list of all feature extractors
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...feature_extraction_utils import FeatureExtractionMixin
from ...image_processing_utils import ImageProcessingMixin
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import TOKENIZER_CONFIG_FILE
from ...utils import FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME, get_file_from_repo, logging
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)
from .feature_extraction_auto import AutoFeatureExtractor
from .image_processing_auto import AutoImageProcessor
from .tokenization_auto import AutoTokenizer


logger = logging.get_logger(__name__)

PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("align", "AlignProcessor"),
        ("altclip", "AltCLIPProcessor"),
        ("bark", "BarkProcessor"),
        ("blip", "BlipProcessor"),
        ("blip-2", "Blip2Processor"),
        ("bridgetower", "BridgeTowerProcessor"),
        ("chameleon", "ChameleonProcessor"),
        ("chinese_clip", "ChineseCLIPProcessor"),
        ("clap", "ClapProcessor"),
        ("clip", "CLIPProcessor"),
        ("clipseg", "CLIPSegProcessor"),
        ("clvp", "ClvpProcessor"),
        ("flava", "FlavaProcessor"),
        ("fuyu", "FuyuProcessor"),
        ("git", "GitProcessor"),
        ("grounding-dino", "GroundingDinoProcessor"), #GroundingDinoProcessor
        ("groupvit", "CLIPProcessor"),
        ("hubert", "Wav2Vec2Processor"),
        ("idefics", "IdeficsProcessor"),
        ("idefics2", "Idefics2Processor"),
        ("idefics3", "Idefics3Processor"),
        ("instructblip", "InstructBlipProcessor"),
        ("instructblipvideo", "InstructBlipVideoProcessor"),
        ("kosmos-2", "Kosmos2Processor"),
        ("layoutlmv2", "LayoutLMv2Processor"),
        ("layoutlmv3", "LayoutLMv3Processor"),
        ("llava", "LlavaProcessor"),
        ("llava_next", "LlavaNextProcessor"),
        ("llava_next_video", "LlavaNextVideoProcessor"),
        ("llava_onevision", "LlavaOnevisionProcessor"),
        ("markuplm", "MarkupLMProcessor"),
        ("mctct", "MCTCTProcessor"),
        ("mgp-str", "MgpstrProcessor"),
        ("mllama", "MllamaProcessor"),
        ("oneformer", "OneFormerProcessor"),
        ("owlv2", "Owlv2Processor"),
        ("owlvit", "OwlViTProcessor"),
        ("paligemma", "PaliGemmaProcessor"),
        ("pix2struct", "Pix2StructProcessor"),
        ("pixtral", "PixtralProcessor"),
        ("pop2piano", "Pop2PianoProcessor"),
        ("qwen2_audio", "Qwen2AudioProcessor"), #处理器依赖qwen2_audio
        ("qwen2_vl", "Qwen2VLProcessor"),
        ("sam", "SamProcessor"),
        ("seamless_m4t", "SeamlessM4TProcessor"),
        ("sew", "Wav2Vec2Processor"),
        ("sew-d", "Wav2Vec2Processor"),
        ("siglip", "SiglipProcessor"),
        ("speech_to_text", "Speech2TextProcessor"),
        ("speech_to_text_2", "Speech2Text2Processor"),
        ("speecht5", "SpeechT5Processor"),
        ("trocr", "TrOCRProcessor"),
        ("tvlt", "TvltProcessor"),
        ("tvp", "TvpProcessor"),
        ("udop", "UdopProcessor"),
        ("unispeech", "Wav2Vec2Processor"),
        ("unispeech-sat", "Wav2Vec2Processor"),
        ("video_llava", "VideoLlavaProcessor"),
        ("vilt", "ViltProcessor"),
        ("vipllava", "LlavaProcessor"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderProcessor"),
        ("wav2vec2", "Wav2Vec2Processor"),
        ("wav2vec2-bert", "Wav2Vec2Processor"),
        ("wav2vec2-conformer", "Wav2Vec2Processor"),
        ("wavlm", "Wav2Vec2Processor"),
        ("whisper", "WhisperProcessor"),
        ("xclip", "XCLIPProcessor"),
    ]
)

PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)


def Instantiation_processor_class(class_name: str):
    for module_name, processors in PROCESSOR_MAPPING_NAMES.items():
        if class_name in processors:
            module_name = model_type_to_module_name(module_name) # str = grounding_dino

            module = importlib.import_module(f".{module_name}", "transformers.models") #+'qwen2_audio'
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for processor in PROCESSOR_MAPPING._extra_content.values():
        if getattr(processor, "__name__", None) == class_name:
            return processor

    # We did not fine the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


# class AutoProcessor:

#     def __init__(self):
#         raise EnvironmentError(
#             "AutoProcessor is designed to be instantiated "
#             "using the `AutoProcessor.from_pretrained(model_name_or_path)` method."
#         )
        
        
#     @classmethod
#     def _handle_deprecated(cls, kwargs):
#         use_auth_token = kwargs.pop("use_auth_token", None)
#         if use_auth_token is not None:
#             warnings.warn(
#                 "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
#                 FutureWarning,
#             )
#             if kwargs.get("token", None) is not None:
#                 raise ValueError(
#                     "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
#                 )
#             kwargs["token"] = use_auth_token
#         return kwargs
    
    
#     @classmethod
#     @replace_list_option_in_docstrings(PROCESSOR_MAPPING_NAMES)
#     def from_pretrained(cls, model_name_or_path, **kwargs):

#         kwargs = cls._handle_deprecated(kwargs)


#         # 获取其他可能的配置和参数
#         config = kwargs.pop("config", None)
#         trust_remote_code = kwargs.pop("trust_remote_code", None)
#         kwargs["_from_auto"] = True  # 设置标志

#         # 定义默认的处理器类和映射
#         processor_class = None
#         processor_auto_map = None


#         """ 第一步 获取processor从processor_config """
#         # 获取用于 repo 文件获取的相关参数
#         get_file_from_repo_kwargs = {
#             key: kwargs[key] for key in inspect.signature(get_file_from_repo).parameters.keys() if key in kwargs
#         }

#         # 检查是否存在处理器配置文件
#         processor_config_file = get_file_from_repo(
#             model_name_or_path, PROCESSOR_NAME, **get_file_from_repo_kwargs
#         )

#         if processor_config_file is not None:
#             config_dict, _ = ProcessorMixin.get_processor_dict(model_name_or_path, **kwargs)
#             processor_class = config_dict.get("processor_class", None)
#             if "AutoProcessor" in config_dict.get("auto_map", {}):
#                 processor_auto_map = config_dict["auto_map"]["AutoProcessor"]



#         """ 第二步 获取processor从其他方法 """
#         # 如果没有找到处理器配置文件，检查其他类型的配置
#         if processor_class is None:
#             # 检查图像处理器配置
#             preprocessor_config_file = get_file_from_repo( #从预处理中拿到
#                 model_name_or_path, FEATURE_EXTRACTOR_NAME, **get_file_from_repo_kwargs
#             )
#             if preprocessor_config_file is not None:
#                 config_dict, _ = ImageProcessingMixin.get_image_processor_dict(model_name_or_path, **kwargs)
#                 processor_class = config_dict.get("processor_class", None)
#                 if "AutoProcessor" in config_dict.get("auto_map", {}):
#                     processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

#             # 检查特征提取器配置
#             if preprocessor_config_file is not None and processor_class is None:
#                 config_dict, _ = FeatureExtractionMixin.get_feature_extractor_dict(
#                     model_name_or_path, **kwargs
#                 )
#                 processor_class = config_dict.get("processor_class", None)
#                 if "AutoProcessor" in config_dict.get("auto_map", {}):
#                     processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

#         # 如果仍然没有找到处理器类，检查 tokenizer 配置
#         if processor_class is None:
#             tokenizer_config_file = get_file_from_repo(
#                 model_name_or_path, TOKENIZER_CONFIG_FILE, **get_file_from_repo_kwargs
#             )
#             if tokenizer_config_file is not None:
#                 with open(tokenizer_config_file, encoding="utf-8") as reader:
#                     config_dict = json.load(reader)

#                 processor_class = config_dict.get("processor_class", None)
#                 if "AutoProcessor" in config_dict.get("auto_map", {}):
#                     processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

#         # 如果仍未找到处理器类，加载 config 并检查是否包含处理器类
#         if processor_class is None:
#             if not isinstance(config, PretrainedConfig):
#                 config = AutoConfig.from_pretrained(
#                     model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
#                 )

#             processor_class = getattr(config, "processor_class", None)
#             if hasattr(config, "auto_map") and "AutoProcessor" in config.auto_map:
#                 processor_auto_map = config.auto_map["AutoProcessor"]

#         # 如果找到了处理器类，通过类名加载它
#         if processor_class is not None:
#             processor_class = Instantiation_processor_class(processor_class)





#         # 判断是否存在远程代码支持
#         has_local_code = processor_class is not None or type(config) in PROCESSOR_MAPPING #true
#         has_remote_code = processor_auto_map is not None #false


#         # 决定是否信任远程代码
#         trust_remote_code = resolve_trust_remote_code( #false
#             trust_remote_code, model_name_or_path, has_local_code, has_remote_code
#         )

#         # 如果有远程代码并且信任，加载远程处理器类
#         if has_remote_code and trust_remote_code:
#             processor_class = get_class_from_dynamic_module(
#                 processor_auto_map, model_name_or_path, **kwargs
#             )
#             _ = kwargs.pop("code_revision", None)
#             if os.path.isdir(model_name_or_path):
#                 processor_class.register_for_auto_class()
#             return processor_class.from_pretrained(
#                 model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
#             )

#         # 如果有本地处理器类，直接加载
#         elif processor_class is not None:
#             return processor_class.from_pretrained(
#                 model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
#             )

#         # 最后，尝试使用 PROCESSOR_MAPPING 加载处理器
#         elif type(config) in PROCESSOR_MAPPING:
#             return PROCESSOR_MAPPING[type(config)].from_pretrained(model_name_or_path, **kwargs)

#         # 如果以上都没有找到，尝试加载 tokenizer、image processor 或 feature extractor
#         try:
#             return AutoTokenizer.from_pretrained(
#                 model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
#             )
#         except Exception:
#             try:
#                 return AutoImageProcessor.from_pretrained(
#                     model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
#                 )
#             except Exception:
#                 pass

#             try:
#                 return AutoFeatureExtractor.from_pretrained(
#                     model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
#                 )
#             except Exception:
#                 pass

#         # 如果仍然没有找到任何处理器类，抛出异常
#         raise ValueError(
#             f"Unrecognized processing class in {model_name_or_path}. Can't instantiate a processor, a "
#             "tokenizer, an image processor or a feature extractor for this model. Make sure the repository contains "
#             "the files of at least one of those processing classes."
#         )



#     @staticmethod
#     def register(config_class, processor_class, exist_ok=False):
#         """
#         Register a new processor for this class.

#         Args:
#             config_class ([`PretrainedConfig`]):
#                 The configuration corresponding to the model to register.
#             processor_class ([`FeatureExtractorMixin`]): The processor to register.
#         """
#         PROCESSOR_MAPPING.register(config_class, processor_class, exist_ok=exist_ok)





class AutoProcessor:

    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the AutoProcessor.from_pretrained(model_name_or_path) method."
        )


    @classmethod
    def _load_processor_config(cls, model_name_or_path, kwargs, get_file_from_repo_kwargs):
        processor_class = None
        processor_auto_map = None

        processor_config_file = get_file_from_repo(
            model_name_or_path, PROCESSOR_NAME, **get_file_from_repo_kwargs #看有没有处理器配置文件
        )
        if processor_config_file is not None:
            config_dict, _ = ProcessorMixin.get_processor_dict(model_name_or_path, **kwargs)
            processor_class = config_dict.get("processor_class", None)
            if "AutoProcessor" in config_dict.get("auto_map", {}):
                processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        return processor_class, processor_auto_map

    @classmethod
    def _load_alternative_configs(cls, model_name_or_path, kwargs, get_file_from_repo_kwargs):
        processor_class = None
        processor_auto_map = None

        preprocessor_config_file = get_file_from_repo( #获取到前处理配置
            model_name_or_path, FEATURE_EXTRACTOR_NAME, **get_file_from_repo_kwargs
        ) #->Qwen/Qwen2-Audio-7B-Instruct-Int4/preprocessor_config.json
        if preprocessor_config_file is not None: #两种实现方式
            config_dict, _ = ImageProcessingMixin.get_image_processor_dict(model_name_or_path, **kwargs) #解析配置文件
            processor_class = config_dict.get("processor_class", None) #'Qwen2AudioProcessor'
            if "AutoProcessor" in config_dict.get("auto_map", {}):
                processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        if preprocessor_config_file is not None and processor_class is None: #文件有类没有
            config_dict, _ = FeatureExtractionMixin.get_feature_extractor_dict(
                model_name_or_path, **kwargs
            )
            processor_class = config_dict.get("processor_class", None)
            if "AutoProcessor" in config_dict.get("auto_map", {}):
                processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        return processor_class, processor_auto_map #获取处理类和映射从字典中

    @classmethod
    def _load_tokenizer_config(cls, model_name_or_path, kwargs, get_file_from_repo_kwargs):
        processor_class = None
        processor_auto_map = None

        tokenizer_config_file = get_file_from_repo(
            model_name_or_path, TOKENIZER_CONFIG_FILE, **get_file_from_repo_kwargs
        )
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding="utf-8") as reader:
                config_dict = json.load(reader)

            processor_class = config_dict.get("processor_class", None)
            if "AutoProcessor" in config_dict.get("auto_map", {}):
                processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        return processor_class, processor_auto_map

    @classmethod
    def _load_from_config(cls, model_name_or_path, kwargs, config, trust_remote_code):
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )

        processor_class = getattr(config, "processor_class", None)
        processor_auto_map = None
        if hasattr(config, "auto_map") and "AutoProcessor" in config.auto_map:
            processor_auto_map = config.auto_map["AutoProcessor"]

        return processor_class, processor_auto_map, config

    @classmethod
    @replace_list_option_in_docstrings(PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, model_name_or_path, **kwargs):

        # 获取其他可能的配置和参数
        config = kwargs.pop("config", None) #理解为参数配置
        trust_remote_code = kwargs.pop("trust_remote_code", None) #none
        kwargs["_from_auto"] = True  # 设置标志

        # 定义默认的处理器类和映射
        get_file_from_repo_kwargs = {key: kwargs[key] for key in inspect.signature(get_file_from_repo).parameters.keys() if key in kwargs}

        # 封装的加载逻辑
        processor_class, processor_auto_map = cls._load_processor_config(
            model_name_or_path, kwargs, get_file_from_repo_kwargs
        )

        if processor_class is None: #使用其他方式获取
            processor_class, processor_auto_map = cls._load_alternative_configs(
                model_name_or_path, kwargs, get_file_from_repo_kwargs
            )

        if processor_class is None:
            processor_class, processor_auto_map = cls._load_tokenizer_config(
                model_name_or_path, kwargs, get_file_from_repo_kwargs
            )

        if processor_class is None:
            processor_class, processor_auto_map, config = cls._load_from_config(
                model_name_or_path, kwargs, config, trust_remote_code
            )

        # 如果找到了处理器类，通过类名加载它
        if processor_class is not None:
            processor_class = Instantiation_processor_class(processor_class) #探寻内存空间

        # 判断是否存在远程代码支持
        has_remote_code = processor_auto_map!=None #没有远程代码
        has_local_code = processor_class!=None or type(config) in PROCESSOR_MAPPING #有本地代码  

        # 决定是否信任远程代码
        trust_remote_code = resolve_trust_remote_code( #本地有就不相信远程了
            trust_remote_code, model_name_or_path, has_local_code, has_remote_code
        )

        # 如果有远程代码并且信任，加载远程处理器类
        if has_remote_code and trust_remote_code: #远程的形式
            processor_class = get_class_from_dynamic_module(
                processor_auto_map, model_name_or_path, **kwargs
            )
            _ = kwargs.pop("code_revision", None)
            if os.path.isdir(model_name_or_path):
                processor_class.register_for_auto_class()
            return processor_class.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )

        # 如果有本地处理器类，直接加载
        elif processor_class is not None:
            return processor_class.from_pretrained( #最重点，处理器类的实例化
                model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )

        # 最后，尝试使用 PROCESSOR_MAPPING 加载处理器
        elif type(config) in PROCESSOR_MAPPING:
            return PROCESSOR_MAPPING[type(config)].from_pretrained(model_name_or_path, **kwargs)

        # 如果以上都没有找到，尝试加载 tokenizer、image processor 或 feature extractor
        try:
            return AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        except Exception:
            try:
                return AutoImageProcessor.from_pretrained(
                    model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            except Exception:
                pass

            try:
                return AutoFeatureExtractor.from_pretrained(
                    model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            except Exception:
                pass

        # 如果仍然没有找到任何处理器类，抛出异常
        raise ValueError(
            f"Unrecognized processing class in {model_name_or_path}. Can't instantiate a processor, a "
            "tokenizer, an image processor or a feature extractor for this model. Make sure the repository contains "
            "the files of at least one of those processing classes."
        )