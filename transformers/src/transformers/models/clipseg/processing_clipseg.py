# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""
Image/Text processor class for CLIPSeg
"""

import warnings

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class CLIPSegProcessor(ProcessorMixin):
    r"""
    Constructs a CLIPSeg processor which wraps a CLIPSeg image processor and a CLIP tokenizer into a single processor.

    [`CLIPSegProcessor`] offers all the functionalities of [`ViTImageProcessor`] and [`CLIPTokenizerFast`]. See the
    [`~CLIPSegProcessor.__call__`] and [`~CLIPSegProcessor.decode`] for more information.

    Args:
        image_processor ([`ViTImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ViTImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(self, text=None, images=None, visual_prompt=None, return_tensors=None, **kwargs):
        """
        Returns:

        """
        if text is None and visual_prompt is None and images is None:
            raise ValueError("You have to specify either text, visual prompt or images.") #总归传入一个吧

        if text is not None and visual_prompt is not None:
            raise ValueError("You have to specify exactly one type of prompt. Either text or visual prompt.") #文本和视觉提示总归传入一个吧

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if visual_prompt is not None:
            prompt_features = self.image_processor(visual_prompt, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)

        if visual_prompt is not None and images is not None:
            encoding = {
                "pixel_values": image_features.pixel_values,
                "conditional_pixel_values": prompt_features.pixel_values,
            }
            return encoding
        elif text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        elif visual_prompt is not None:
            encoding = {
                "conditional_pixel_values": prompt_features.pixel_values,
            }
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
