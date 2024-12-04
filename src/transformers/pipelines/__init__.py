# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from huggingface_hub import model_info

from ..configuration_utils import PretrainedConfig
from ..dynamic_module_utils import get_class_from_dynamic_module
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..models.auto.configuration_auto import AutoConfig
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
from ..models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING, AutoImageProcessor
from ..models.auto.modeling_auto import AutoModelForDepthEstimation, AutoModelForImageToImage
from ..models.auto.processing_auto import PROCESSOR_MAPPING, AutoProcessor
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from ..processing_utils import ProcessorMixin
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
    CONFIG_NAME,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    cached_file,
    extract_commit_hash,
    find_adapter_config_file,
    is_kenlm_available,
    is_offline_mode,
    is_peft_available,
    is_pyctcdecode_available,
    is_tf_available,
    is_torch_available,
    logging,
)
from .audio_classification import AudioClassificationPipeline
from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from .base import (
    ArgumentHandler,
    CsvPipelineDataFormat,
    JsonPipelineDataFormat,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    PipelineException,
    PipelineRegistry,
    get_default_model_and_revision,
    infer_framework_load_model,
)
from .depth_estimation import DepthEstimationPipeline
from .document_question_answering import DocumentQuestionAnsweringPipeline
from .feature_extraction import FeatureExtractionPipeline
from .fill_mask import FillMaskPipeline
from .image_classification import ImageClassificationPipeline
from .image_feature_extraction import ImageFeatureExtractionPipeline
from .image_segmentation import ImageSegmentationPipeline
from .image_text_to_text import ImageTextToTextPipeline
from .image_to_image import ImageToImagePipeline
from .image_to_text import ImageToTextPipeline
from .mask_generation import MaskGenerationPipeline
from .object_detection import ObjectDetectionPipeline
from .question_answering import QuestionAnsweringArgumentHandler, QuestionAnsweringPipeline
from .table_question_answering import TableQuestionAnsweringArgumentHandler, TableQuestionAnsweringPipeline
from .text2text_generation import SummarizationPipeline, Text2TextGenerationPipeline, TranslationPipeline
from .text_classification import TextClassificationPipeline
from .text_generation import TextGenerationPipeline
from .text_to_audio import TextToAudioPipeline
from .token_classification import (
    AggregationStrategy,
    NerPipeline,
    TokenClassificationArgumentHandler,
    TokenClassificationPipeline,
)
from .video_classification import VideoClassificationPipeline
from .visual_question_answering import VisualQuestionAnsweringPipeline
from .zero_shot_audio_classification import ZeroShotAudioClassificationPipeline
from .zero_shot_classification import ZeroShotClassificationArgumentHandler, ZeroShotClassificationPipeline
from .zero_shot_image_classification import ZeroShotImageClassificationPipeline
from .zero_shot_object_detection import ZeroShotObjectDetectionPipeline


if is_tf_available(): #tensorflow的模型
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import (
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForImageClassification,
        TFAutoModelForMaskedLM,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTableQuestionAnswering,
        TFAutoModelForTokenClassification,
        TFAutoModelForVision2Seq,
        TFAutoModelForZeroShotImageClassification,
    )

if is_torch_available(): #pytorch模型
    import torch

    from ..models.auto.modeling_auto import (
        AutoModel,
        AutoModelForAudioClassification,
        AutoModelForCausalLM,
        AutoModelForCTC,
        AutoModelForDocumentQuestionAnswering,
        AutoModelForImageClassification,
        AutoModelForImageSegmentation,
        AutoModelForImageTextToText,
        AutoModelForMaskedLM,
        AutoModelForMaskGeneration,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        AutoModelForSemanticSegmentation,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,
        AutoModelForTableQuestionAnswering,
        AutoModelForTextToSpectrogram,
        AutoModelForTextToWaveform,
        AutoModelForTokenClassification,
        AutoModelForVideoClassification,
        AutoModelForVision2Seq,
        AutoModelForVisualQuestionAnswering,
        AutoModelForZeroShotImageClassification,
        AutoModelForZeroShotObjectDetection,
    )


if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel
    from ..tokenization_utils_fast import PreTrainedTokenizerFast


logger = logging.get_logger(__name__)

#在此注册所有支持的任务
# Register all the supported tasks here 
TASK_ALIASES = { #任务重命名
    "sentiment-analysis": "text-classification",
    "ner": "token-classification", #Named Entity Recognition  实体识别
    "vqa": "visual-question-answering", #视觉问答
    "text-to-speech": "text-to-audio", #文本转语音
}
SUPPORTED_TASKS = {
    "audio-classification": { #音频分类
        "impl": AudioClassificationPipeline,
        "tf": (),
        "pt": (AutoModelForAudioClassification,) if is_torch_available() else (),
        "default": {"model": {"pt": ("superb/wav2vec2-base-superb-ks", "372e048")}},
        "type": "audio",
    },
    "automatic-speech-recognition": { #自动语音识别 (ASR)
        "impl": AutomaticSpeechRecognitionPipeline,
        "tf": (),
        "pt": (AutoModelForCTC, AutoModelForSpeechSeq2Seq) if is_torch_available() else (),
        "default": {"model": {"pt": ("facebook/wav2vec2-base-960h", "22aad52")}},
        "type": "multimodal",
    },
    "text-to-audio": { #文本生成音频
        "impl": TextToAudioPipeline,
        "tf": (),
        "pt": (AutoModelForTextToWaveform, AutoModelForTextToSpectrogram) if is_torch_available() else (),
        "default": {"model": {"pt": ("suno/bark-small", "1dbd7a1")}},
        "type": "text",
    },
    "feature-extraction": { #特征提取
        "impl": FeatureExtractionPipeline,
        "tf": (TFAutoModel,) if is_tf_available() else (),
        "pt": (AutoModel,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("distilbert/distilbert-base-cased", "6ea8117"),
                "tf": ("distilbert/distilbert-base-cased", "6ea8117"),
            }
        },
        "type": "multimodal",
    },
    "text-classification": { #文本分类
        "impl": TextClassificationPipeline,
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "714eb0f"),
                "tf": ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "714eb0f"),
            },
        },
        "type": "text",
    },
    "token-classification": { #标记分类
        "impl": TokenClassificationPipeline,
        "tf": (TFAutoModelForTokenClassification,) if is_tf_available() else (),
        "pt": (AutoModelForTokenClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("dbmdz/bert-large-cased-finetuned-conll03-english", "4c53496"),
                "tf": ("dbmdz/bert-large-cased-finetuned-conll03-english", "4c53496"),
            },
        },
        "type": "text",
    },
    "question-answering": { #问答系统
        "impl": QuestionAnsweringPipeline,
        "tf": (TFAutoModelForQuestionAnswering,) if is_tf_available() else (),
        "pt": (AutoModelForQuestionAnswering,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("distilbert/distilbert-base-cased-distilled-squad", "564e9b5"),
                "tf": ("distilbert/distilbert-base-cased-distilled-squad", "564e9b5"),
            },
        },
        "type": "text",
    },
    "table-question-answering": { #表格问答
        "impl": TableQuestionAnsweringPipeline,
        "pt": (AutoModelForTableQuestionAnswering,) if is_torch_available() else (),
        "tf": (TFAutoModelForTableQuestionAnswering,) if is_tf_available() else (),
        "default": {
            "model": {
                "pt": ("google/tapas-base-finetuned-wtq", "e3dde19"),
                "tf": ("google/tapas-base-finetuned-wtq", "e3dde19"),
            },
        },
        "type": "text",
    },
    "visual-question-answering": { #视觉问答
        "impl": VisualQuestionAnsweringPipeline,
        "pt": (AutoModelForVisualQuestionAnswering,) if is_torch_available() else (),
        "tf": (),
        "default": {
            "model": {"pt": ("dandelin/vilt-b32-finetuned-vqa", "d0a1f6a")},
        },
        "type": "multimodal",
    },
    "document-question-answering": { #文档问答
        "impl": DocumentQuestionAnsweringPipeline,
        "pt": (AutoModelForDocumentQuestionAnswering,) if is_torch_available() else (),
        "tf": (),
        "default": {
            "model": {"pt": ("impira/layoutlm-document-qa", "beed3c4")},
        },
        "type": "multimodal",
    },
    "fill-mask": { #掩码填充
        "impl": FillMaskPipeline,
        "tf": (TFAutoModelForMaskedLM,) if is_tf_available() else (),
        "pt": (AutoModelForMaskedLM,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("distilbert/distilroberta-base", "fb53ab8"),
                "tf": ("distilbert/distilroberta-base", "fb53ab8"),
            }
        },
        "type": "text",
    },
    "summarization": { #文本摘要
        "impl": SummarizationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {
            "model": {"pt": ("sshleifer/distilbart-cnn-12-6", "a4f8f3e"), "tf": ("google-t5/t5-small", "df1b051")}
        },
        "type": "text",
    },
    # This task is a special case as it's parametrized by SRC, TGT languages.
    "translation": { #机器翻译
        "impl": TranslationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {
            ("en", "fr"): {"model": {"pt": ("google-t5/t5-base", "a9723ea"), "tf": ("google-t5/t5-base", "a9723ea")}},
            ("en", "de"): {"model": {"pt": ("google-t5/t5-base", "a9723ea"), "tf": ("google-t5/t5-base", "a9723ea")}},
            ("en", "ro"): {"model": {"pt": ("google-t5/t5-base", "a9723ea"), "tf": ("google-t5/t5-base", "a9723ea")}},
        },
        "type": "text",
    },
    "text2text-generation": { #文本生成
        "impl": Text2TextGenerationPipeline,
        "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
        "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
        "default": {"model": {"pt": ("google-t5/t5-base", "a9723ea"), "tf": ("google-t5/t5-base", "a9723ea")}},
        "type": "text",
    },
    "text-generation": { #语言生成
        "impl": TextGenerationPipeline,
        "tf": (TFAutoModelForCausalLM,) if is_tf_available() else (),
        "pt": (AutoModelForCausalLM,) if is_torch_available() else (),
        "default": {"model": {"pt": ("openai-community/gpt2", "607a30d"), "tf": ("openai-community/gpt2", "607a30d")}},
        "type": "text",
    },
    "zero-shot-classification": { #零样本分类
        "impl": ZeroShotClassificationPipeline,
        "tf": (TFAutoModelForSequenceClassification,) if is_tf_available() else (),
        "pt": (AutoModelForSequenceClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("facebook/bart-large-mnli", "d7645e1"),
                "tf": ("FacebookAI/roberta-large-mnli", "2a8f12d"),
            },
            "config": {
                "pt": ("facebook/bart-large-mnli", "d7645e1"),
                "tf": ("FacebookAI/roberta-large-mnli", "2a8f12d"),
            },
        },
        "type": "text",
    },
    "zero-shot-image-classification": { #零样本图像分类
        "impl": ZeroShotImageClassificationPipeline,
        "tf": (TFAutoModelForZeroShotImageClassification,) if is_tf_available() else (),
        "pt": (AutoModelForZeroShotImageClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("openai/clip-vit-base-patch32", "3d74acf"),
                "tf": ("openai/clip-vit-base-patch32", "3d74acf"),
            }
        },
        "type": "multimodal",
    },
    "zero-shot-audio-classification": { #零样本音频分类
        "impl": ZeroShotAudioClassificationPipeline,
        "tf": (),
        "pt": (AutoModel,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("laion/clap-htsat-fused", "cca9e28"),
            }
        },
        "type": "multimodal",
    },
    "image-classification": { #图像分类
        "impl": ImageClassificationPipeline,
        "tf": (TFAutoModelForImageClassification,) if is_tf_available() else (),
        "pt": (AutoModelForImageClassification,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("google/vit-base-patch16-224", "3f49326"),
                "tf": ("google/vit-base-patch16-224", "3f49326"),
            }
        },
        "type": "image",
    },
    "image-feature-extraction": { #图像特征提取
        "impl": ImageFeatureExtractionPipeline,
        "tf": (TFAutoModel,) if is_tf_available() else (),
        "pt": (AutoModel,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("google/vit-base-patch16-224", "3f49326"),
                "tf": ("google/vit-base-patch16-224", "3f49326"),
            }
        },
        "type": "image",
    },
    "image-segmentation": { #图像分割
        "impl": ImageSegmentationPipeline,
        "tf": (),
        "pt": (AutoModelForImageSegmentation, AutoModelForSemanticSegmentation) if is_torch_available() else (),
        "default": {"model": {"pt": ("facebook/detr-resnet-50-panoptic", "d53b52a")}},
        "type": "multimodal",
    },
    "image-to-text": { #图像转文本
        "impl": ImageToTextPipeline,
        "tf": (TFAutoModelForVision2Seq,) if is_tf_available() else (),
        "pt": (AutoModelForVision2Seq,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("ydshieh/vit-gpt2-coco-en", "5bebf1e"),
                "tf": ("ydshieh/vit-gpt2-coco-en", "5bebf1e"),
            }
        },
        "type": "multimodal",
    },
    "image-text-to-text": { #图像文本混合生成文本
        "impl": ImageTextToTextPipeline,
        "tf": (),
        "pt": (AutoModelForImageTextToText,) if is_torch_available() else (),
        "default": {
            "model": {
                "pt": ("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "2c9ba3b"),
            }
        },
        "type": "multimodal",
    },
    "object-detection": { #目标检测
        "impl": ObjectDetectionPipeline,
        "tf": (),
        "pt": (AutoModelForObjectDetection,) if is_torch_available() else (),
        "default": {"model": {"pt": ("facebook/detr-resnet-50", "1d5f47b")}},
        "type": "multimodal",
    },
    "zero-shot-object-detection": { #零样本目标检测
        "impl": ZeroShotObjectDetectionPipeline,
        "tf": (),
        "pt": (AutoModelForZeroShotObjectDetection,) if is_torch_available() else (),
        "default": {"model": {"pt": ("google/owlvit-base-patch32", "cbc355f")}},
        "type": "multimodal",
    },
    "depth-estimation": { #深度估计
        "impl": DepthEstimationPipeline,
        "tf": (),
        "pt": (AutoModelForDepthEstimation,) if is_torch_available() else (),
        "default": {"model": {"pt": ("Intel/dpt-large", "bc15f29")}},
        "type": "image",
    },
    "video-classification": { #视频分类
        "impl": VideoClassificationPipeline,
        "tf": (),
        "pt": (AutoModelForVideoClassification,) if is_torch_available() else (),
        "default": {"model": {"pt": ("MCG-NJU/videomae-base-finetuned-kinetics", "488eb9a")}},
        "type": "video",
    },
    "mask-generation": { #掩码生成
        "impl": MaskGenerationPipeline,
        "tf": (),
        "pt": (AutoModelForMaskGeneration,) if is_torch_available() else (),
        "default": {"model": {"pt": ("facebook/sam-vit-huge", "87aecf0")}},
        "type": "multimodal",
    },
    "image-to-image": { #图像到图像生成
        "impl": ImageToImagePipeline,
        "tf": (),
        "pt": (AutoModelForImageToImage,) if is_torch_available() else (),
        "default": {"model": {"pt": ("caidas/swin2SR-classical-sr-x2-64", "cee1c92")}},
        "type": "image",
    },
}

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_IMAGE_PROCESSOR_TASKS = set()
NO_TOKENIZER_TASKS = set()

# Those model configs are special, they are generic over their task, meaning
# any tokenizer/feature_extractor might be use for a given model so we cannot
# use the statically defined TOKENIZER_MAPPING and FEATURE_EXTRACTOR_MAPPING to
# see if the model defines such objects or not.
MULTI_MODEL_AUDIO_CONFIGS = {"SpeechEncoderDecoderConfig"}
MULTI_MODEL_VISION_CONFIGS = {"VisionEncoderDecoderConfig", "VisionTextDualEncoderConfig"}
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values["type"] in {"image", "video"}:
        NO_TOKENIZER_TASKS.add(task)
    elif values["type"] in {"audio"}:
        NO_TOKENIZER_TASKS.add(task)
        NO_IMAGE_PROCESSOR_TASKS.add(task)
    elif values["type"] != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values['type']}")
#工作流实例化
PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)


def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    return PIPELINE_REGISTRY.get_supported_tasks()


def get_task(model: str, token: Optional[str] = None, **deprecated_kwargs) -> str:
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    if is_offline_mode():
        raise RuntimeError("You cannot infer task automatically within `pipeline` when using offline mode")
    try:
        info = model_info(model, token=token)
    except Exception as e:
        raise RuntimeError(f"Instantiating a pipeline without a task set raised an error: {e}")
    if not info.pipeline_tag:
        raise RuntimeError(
            f"The model {model} does not seem to have a correct `pipeline_tag` set to infer the task automatically"
        )
    if getattr(info, "library_name", "transformers") != "transformers":
        raise RuntimeError(f"This model is meant to be used with {info.library_name} not with transformers")
    task = info.pipeline_tag
    return task


def check_task(task: str) -> Tuple[str, Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`
            - `"automatic-speech-recognition"`
            - `"conversational"`
            - `"depth-estimation"`
            - `"document-question-answering"`
            - `"feature-extraction"`
            - `"fill-mask"`
            - `"image-classification"`
            - `"image-feature-extraction"`
            - `"image-segmentation"`
            - `"image-to-text"`
            - `"image-to-image"`
            - `"object-detection"`
            - `"question-answering"`
            - `"summarization"`
            - `"table-question-answering"`
            - `"text2text-generation"`
            - `"text-classification"` (alias `"sentiment-analysis"` available)
            - `"text-generation"`
            - `"text-to-audio"` (alias `"text-to-speech"` available)
            - `"token-classification"` (alias `"ner"` available)
            - `"translation"`
            - `"translation_xx_to_yy"`
            - `"video-classification"`
            - `"visual-question-answering"` (alias `"vqa"` available)
            - `"zero-shot-classification"`
            - `"zero-shot-image-classification"`
            - `"zero-shot-object-detection"`

    Returns:
        (normalized_task: `str`, task_defaults: `dict`, task_options: (`tuple`, None)) The normalized task name
        (removed alias and options). The actual dictionary required to initialize the pipeline and some extra task
        options for parametrized tasks like "translation_XX_to_YY"


    """
    return PIPELINE_REGISTRY.check_task(task)


def clean_custom_task(task_info):
    import transformers

    if "impl" not in task_info:
        raise RuntimeError("This model introduces a custom pipeline without specifying its implementation.")
    pt_class_names = task_info.get("pt", ())
    if isinstance(pt_class_names, str):
        pt_class_names = [pt_class_names]
    task_info["pt"] = tuple(getattr(transformers, c) for c in pt_class_names)
    tf_class_names = task_info.get("tf", ())
    if isinstance(tf_class_names, str):
        tf_class_names = [tf_class_names]
    task_info["tf"] = tuple(getattr(transformers, c) for c in tf_class_names)
    return task_info, None


def pipeline(
    task: str = None,
    model: Optional[Union[str, "PreTrainedModel", "TFPreTrainedModel"]] = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, "PreTrainedTokenizerFast"]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    image_processor: Optional[Union[str, BaseImageProcessor]] = None,
    processor: Optional[Union[str, ProcessorMixin]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    use_fast: bool = True,
    token: Optional[Union[str, bool]] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    device_map=None,
    torch_dtype=None,
    trust_remote_code: Optional[bool] = None,
    model_kwargs: Dict[str, Any] = None,
    pipeline_class: Optional[Any] = None,
    **kwargs,
) -> Pipeline:
    """
    ```python
    >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

    >>> # Sentiment analysis pipeline
    >>> analyzer = pipeline("sentiment-analysis")

    >>> # Question answering pipeline, specifying the checkpoint identifier
    >>> oracle = pipeline(
    ...     "question-answering", model="distilbert/distilbert-base-cased-distilled-squad", tokenizer="google-bert/bert-base-cased"
    ... )

    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
    >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
    ```"""
    if model_kwargs is None:
        model_kwargs = {}
    # Make sure we only pass use_auth_token once as a kwarg (it used to be possible to pass it in model_kwargs,
    # this is to keep BC).
    use_auth_token = model_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    code_revision = kwargs.pop("code_revision", None)
    commit_hash = kwargs.pop("_commit_hash", None)

    hub_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": commit_hash,
    }

    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model "
            "being specified. "
            "Please provide a task class or a model"
        )

    if model is None and tokenizer is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided tokenizer"
            " may not be compatible with the default model. Please provide a PreTrainedModel class or a"
            " path/identifier to a pretrained model when providing tokenizer."
        )
    if model is None and feature_extractor is not None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided"
            " feature_extractor may not be compatible with the default model. Please provide a PreTrainedModel class"
            " or a path/identifier to a pretrained model when providing feature_extractor."
        )
    if isinstance(model, Path):
        model = str(model)

    if commit_hash is None:
        pretrained_model_name_or_path = None
        if isinstance(config, str):
            pretrained_model_name_or_path = config
        elif config is None and isinstance(model, str):
            pretrained_model_name_or_path = model #赋值

        if not isinstance(config, PretrainedConfig) and pretrained_model_name_or_path is not None:
            # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
            resolved_config_file = cached_file( #模型配置文件
                pretrained_model_name_or_path,
                CONFIG_NAME,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                cache_dir=model_kwargs.get("cache_dir"),
                **hub_kwargs,
            )
            hub_kwargs["_commit_hash"] = extract_commit_hash(resolved_config_file, commit_hash)
        else:
            hub_kwargs["_commit_hash"] = getattr(config, "_commit_hash", None)

    # Config is the primordial information item.
    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(
            config, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        hub_kwargs["_commit_hash"] = config._commit_hash
    elif config is None and isinstance(model, str):
        # Check for an adapter file in the model path if PEFT is available
        if is_peft_available():
            # `find_adapter_config_file` doesn't accept `trust_remote_code`
            _hub_kwargs = {k: v for k, v in hub_kwargs.items() if k != "trust_remote_code"}
            maybe_adapter_path = find_adapter_config_file(
                model,
                token=hub_kwargs["token"],
                revision=hub_kwargs["revision"],
                _commit_hash=hub_kwargs["_commit_hash"],
            )

            if maybe_adapter_path is not None:
                with open(maybe_adapter_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)
                    model = adapter_config["base_model_name_or_path"]

        config = AutoConfig.from_pretrained(
            model, _from_pipeline=task, code_revision=code_revision, **hub_kwargs, **model_kwargs
        )
        hub_kwargs["_commit_hash"] = config._commit_hash

    custom_tasks = {}
    if config is not None and len(getattr(config, "custom_pipelines", {})) > 0:
        custom_tasks = config.custom_pipelines
        if task is None and trust_remote_code is not False:
            if len(custom_tasks) == 1:
                task = list(custom_tasks.keys())[0]
            else:
                raise RuntimeError(
                    "We can't infer the task automatically for this model as there are multiple tasks available. Pick "
                    f"one in {', '.join(custom_tasks.keys())}"
                )

    if task is None and model is not None:
        if not isinstance(model, str):
            raise RuntimeError(
                "Inferring the task automatically requires to check the hub with a model_id defined as a `str`. "
                f"{model} is not a valid model_id."
            )
        task = get_task(model, token)

    # Retrieve the task
    if task in custom_tasks:
        normalized_task = task
        targeted_task, task_options = clean_custom_task(custom_tasks[task])
        if pipeline_class is None:
            if not trust_remote_code:
                raise ValueError(
                    "Loading this pipeline requires you to execute the code in the pipeline file in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    " set the option `trust_remote_code=True` to remove this error."
                )
            class_ref = targeted_task["impl"]
            pipeline_class = get_class_from_dynamic_module(
                class_ref,
                model,
                code_revision=code_revision,
                **hub_kwargs,
            )
    else:
        normalized_task, targeted_task, task_options = check_task(task)
        if pipeline_class is None:
            pipeline_class = targeted_task["impl"] #返回工作流类

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        # At that point framework might still be undetermined
        model, default_revision = get_default_model_and_revision(targeted_task, framework, task_options)
        revision = revision if revision is not None else default_revision
        logger.warning(
            f"No model was supplied, defaulted to {model} and revision"
            f" {revision} ({HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{model}).\n"
            "Using a pipeline without specifying a model name and revision in production is not recommended."
        )
        hub_kwargs["revision"] = revision
        if config is None and isinstance(model, str):
            config = AutoConfig.from_pretrained(model, _from_pipeline=task, **hub_kwargs, **model_kwargs)
            hub_kwargs["_commit_hash"] = config._commit_hash

    if device_map is not None:
        if "device_map" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... device_map=..., model_kwargs={"device_map":...})` as those'
                " arguments might conflict, use only one.)"
            )
        if device is not None:
            logger.warning(
                "Both `device` and `device_map` are specified. `device` will override `device_map`. You"
                " will most likely encounter unexpected behavior. Please remove `device` and keep `device_map`."
            )
        model_kwargs["device_map"] = device_map
    if torch_dtype is not None:
        if "torch_dtype" in model_kwargs:
            raise ValueError(
                'You cannot use both `pipeline(... torch_dtype=..., model_kwargs={"torch_dtype":...})` as those'
                " arguments might conflict, use only one.)"
            )
        if isinstance(torch_dtype, str) and hasattr(torch, torch_dtype):
            torch_dtype = getattr(torch, torch_dtype)
        model_kwargs["torch_dtype"] = torch_dtype

    model_name = model if isinstance(model, str) else None

    # Load the correct model if possible
    # Infer the framework from the model if not already defined
    if isinstance(model, str) or framework is None:
        model_classes = {"tf": targeted_task["tf"], "pt": targeted_task["pt"]} #模型类
        framework, model = infer_framework_load_model(
            model, #模型名
            model_classes=model_classes, #模型类
            config=config, #配置类
            framework=framework, #none
            task=task, #任务名
            **hub_kwargs,
            **model_kwargs,
        )

    model_config = model.config
    hub_kwargs["_commit_hash"] = model.config._commit_hash

    load_tokenizer = type(model_config) in TOKENIZER_MAPPING or model_config.tokenizer_class is not None
    load_feature_extractor = type(model_config) in FEATURE_EXTRACTOR_MAPPING or feature_extractor is not None
    load_image_processor = type(model_config) in IMAGE_PROCESSOR_MAPPING or image_processor is not None
    load_processor = type(model_config) in PROCESSOR_MAPPING or processor is not None

    # Check that pipeline class required loading
    load_tokenizer = load_tokenizer and pipeline_class._load_tokenizer
    load_feature_extractor = load_feature_extractor and pipeline_class._load_feature_extractor
    load_image_processor = load_image_processor and pipeline_class._load_image_processor
    load_processor = load_processor and pipeline_class._load_processor

    # If `model` (instance of `PretrainedModel` instead of `str`) is passed (and/or same for config), while
    # `image_processor` or `feature_extractor` is `None`, the loading will fail. This happens particularly for some
    # vision tasks when calling `pipeline()` with `model` and only one of the `image_processor` and `feature_extractor`.
    # TODO: we need to make `NO_IMAGE_PROCESSOR_TASKS` and `NO_FEATURE_EXTRACTOR_TASKS` more robust to avoid such issue.
    # This block is only temporarily to make CI green.
    if load_image_processor and load_feature_extractor:
        load_feature_extractor = False

    if (
        tokenizer is None
        and not load_tokenizer
        and normalized_task not in NO_TOKENIZER_TASKS
        # Using class name to avoid importing the real class.
        and (
            model_config.__class__.__name__ in MULTI_MODEL_AUDIO_CONFIGS
            or model_config.__class__.__name__ in MULTI_MODEL_VISION_CONFIGS
        )
    ):
        # This is a special category of models, that are fusions of multiple models
        # so the model_config might not define a tokenizer, but it seems to be
        # necessary for the task, so we're force-trying to load it.
        load_tokenizer = True
    if (
        image_processor is None
        and not load_image_processor
        and normalized_task not in NO_IMAGE_PROCESSOR_TASKS
        # Using class name to avoid importing the real class.
        and model_config.__class__.__name__ in MULTI_MODEL_VISION_CONFIGS
    ):
        # This is a special category of models, that are fusions of multiple models
        # so the model_config might not define a tokenizer, but it seems to be
        # necessary for the task, so we're force-trying to load it.
        load_image_processor = True
    if (
        feature_extractor is None
        and not load_feature_extractor
        and normalized_task not in NO_FEATURE_EXTRACTOR_TASKS
        # Using class name to avoid importing the real class.
        and model_config.__class__.__name__ in MULTI_MODEL_AUDIO_CONFIGS
    ):
        # This is a special category of models, that are fusions of multiple models
        # so the model_config might not define a tokenizer, but it seems to be
        # necessary for the task, so we're force-trying to load it.
        load_feature_extractor = True

    if task in NO_TOKENIZER_TASKS:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False

    if task in NO_FEATURE_EXTRACTOR_TASKS:
        load_feature_extractor = False
    if task in NO_IMAGE_PROCESSOR_TASKS:
        load_image_processor = False

    if load_tokenizer:
        # Try to infer tokenizer from model or config name (if provided as str)
        if tokenizer is None:
            if isinstance(model_name, str):
                tokenizer = model_name
            elif isinstance(config, str):
                tokenizer = config
            else:
                # Impossible to guess what is the right tokenizer here
                raise Exception(
                    "Impossible to guess which tokenizer to use. "
                    "Please provide a PreTrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                )

        # Instantiate tokenizer if needed
        if isinstance(tokenizer, (str, tuple)):
            if isinstance(tokenizer, tuple):
                # For tuple we have (tokenizer name, {kwargs})
                use_fast = tokenizer[1].pop("use_fast", use_fast)
                tokenizer_identifier = tokenizer[0]
                tokenizer_kwargs = tokenizer[1]
            else:
                tokenizer_identifier = tokenizer
                tokenizer_kwargs = model_kwargs.copy()
                tokenizer_kwargs.pop("torch_dtype", None)

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_identifier, use_fast=use_fast, _from_pipeline=task, **hub_kwargs, **tokenizer_kwargs
            )

    if load_image_processor:
        # Try to infer image processor from model or config name (if provided as str)
        if image_processor is None:
            if isinstance(model_name, str):
                image_processor = model_name
            elif isinstance(config, str):
                image_processor = config
            # Backward compatibility, as `feature_extractor` used to be the name
            # for `ImageProcessor`.
            elif feature_extractor is not None and isinstance(feature_extractor, BaseImageProcessor):
                image_processor = feature_extractor
            else:
                # Impossible to guess what is the right image_processor here
                raise Exception(
                    "Impossible to guess which image processor to use. "
                    "Please provide a PreTrainedImageProcessor class or a path/identifier "
                    "to a pretrained image processor."
                )

        # Instantiate image_processor if needed
        if isinstance(image_processor, (str, tuple)):
            image_processor = AutoImageProcessor.from_pretrained(
                image_processor, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )

    if load_feature_extractor:
        # Try to infer feature extractor from model or config name (if provided as str)
        if feature_extractor is None:
            if isinstance(model_name, str):
                feature_extractor = model_name
            elif isinstance(config, str):
                feature_extractor = config
            else:
                # Impossible to guess what is the right feature_extractor here
                raise Exception(
                    "Impossible to guess which feature extractor to use. "
                    "Please provide a PreTrainedFeatureExtractor class or a path/identifier "
                    "to a pretrained feature extractor."
                )

        # Instantiate feature_extractor if needed
        if isinstance(feature_extractor, (str, tuple)):
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                feature_extractor, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )

            if (
                feature_extractor._processor_class
                and feature_extractor._processor_class.endswith("WithLM")
                and isinstance(model_name, str)
            ):
                try:
                    import kenlm  # to trigger `ImportError` if not installed
                    from pyctcdecode import BeamSearchDecoderCTC

                    if os.path.isdir(model_name) or os.path.isfile(model_name):
                        decoder = BeamSearchDecoderCTC.load_from_dir(model_name)
                    else:
                        language_model_glob = os.path.join(
                            BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY, "*"
                        )
                        alphabet_filename = BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
                        allow_patterns = [language_model_glob, alphabet_filename]
                        decoder = BeamSearchDecoderCTC.load_from_hf_hub(model_name, allow_patterns=allow_patterns)

                    kwargs["decoder"] = decoder
                except ImportError as e:
                    logger.warning(f"Could not load the `decoder` for {model_name}. Defaulting to raw CTC. Error: {e}")
                    if not is_kenlm_available():
                        logger.warning("Try to install `kenlm`: `pip install kenlm")

                    if not is_pyctcdecode_available():
                        logger.warning("Try to install `pyctcdecode`: `pip install pyctcdecode")

    if load_processor:
        # Try to infer processor from model or config name (if provided as str)
        if processor is None:
            if isinstance(model_name, str):
                processor = model_name
            elif isinstance(config, str):
                processor = config
            else:
                # Impossible to guess what is the right processor here
                raise Exception(
                    "Impossible to guess which processor to use. "
                    "Please provide a processor instance or a path/identifier "
                    "to a processor."
                )

        # Instantiate processor if needed
        if isinstance(processor, (str, tuple)):
            processor = AutoProcessor.from_pretrained(processor, _from_pipeline=task, **hub_kwargs, **model_kwargs)
            if not isinstance(processor, ProcessorMixin):
                raise TypeError(
                    "Processor was loaded, but it is not an instance of `ProcessorMixin`. "
                    f"Got type `{type(processor)}` instead. Please check that you specified "
                    "correct pipeline task for the model and model has processor implemented and saved."
                )

    if task == "translation" and model.config.task_specific_params:
        for key in model.config.task_specific_params:
            if key.startswith("translation"):
                task = key
                warnings.warn(
                    f'"translation" task was used, instead of "translation_XX_to_YY", defaulting to "{task}"',
                    UserWarning,
                )
                break

    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer

    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor

    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    if image_processor is not None:
        kwargs["image_processor"] = image_processor

    if device is not None:
        kwargs["device"] = device

    if processor is not None:
        kwargs["processor"] = processor

    return pipeline_class(model=model, framework=framework, task=task, **kwargs)
