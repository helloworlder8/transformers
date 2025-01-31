# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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


import copy
import json
import os
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .utils import (
    IMAGE_PROCESSOR_NAME,
    PushToHubMixin,
    add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines,
    cached_file,
    copy_func,
    download_url,
    is_offline_mode,
    is_remote_url,
    is_vision_available,
    logging,
)


if is_vision_available():
    from PIL import Image


logger = logging.get_logger(__name__)


# TODO: Move BatchFeature to be imported by both image_processing_utils and image_processing_utils
# We override the class string here, but logic is the same.
class BatchFeature(BaseBatchFeature):
    r"""
    Holds the output of the image processor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """


# TODO: (Amy) - factor out the common parts of this and the feature extractor
class ImageProcessingMixin(PushToHubMixin):
    """
    This is an image processor mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """

    _auto_class = None

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # This key was saved while we still used `XXXFeatureExtractor` for image processing. Now we use
        # `XXXImageProcessor`, this attribute and its value are misleading.
        kwargs.pop("feature_extractor_type", None)
        # Pop "processor_class" as it should be saved as private attribute
        self._processor_class = kwargs.pop("processor_class", None)
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    @classmethod
    def from_pretrained(
        cls, #类
        model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):

        kwargs["cache_dir"] = cache_dir #缓存路径
        kwargs["force_download"] = force_download #强制下载
        kwargs["local_files_only"] = local_files_only #只要本地文件
        kwargs["revision"] = revision #修订版本

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        image_processor_dict, kwargs = cls.get_image_processor_dict(model_name_or_path, **kwargs) #新生成的字典

        return cls.from_dict(image_processor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save an image processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~image_processing_utils.ImageProcessingMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the image processor JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_image_processor_file = os.path.join(save_directory, IMAGE_PROCESSOR_NAME)

        self.to_json_file(output_image_processor_file)
        logger.info(f"Image processor saved in {output_image_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return [output_image_processor_file]

    # @classmethod
    # def get_image_processor_dict(
    #     cls, model_name_or_path: Union[str, os.PathLike], **kwargs
    # ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    #     """
    #     From a `model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
    #     image processor of type [`~image_processor_utils.ImageProcessingMixin`] using `from_dict`.

    #     Parameters:
    #         model_name_or_path (`str` or `os.PathLike`):
    #             The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
    #         subfolder (`str`, *optional*, defaults to `""`):
    #             In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
    #             specify the folder name here.

    #     Returns:
    #         `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the image processor object.
    #     """
    #     cache_dir = kwargs.pop("cache_dir", None)
    #     force_download = kwargs.pop("force_download", False)
    #     resume_download = kwargs.pop("resume_download", None)
    #     proxies = kwargs.pop("proxies", None)
    #     token = kwargs.pop("token", None)
    #     local_files_only = kwargs.pop("local_files_only", False)
    #     revision = kwargs.pop("revision", None)
    #     subfolder = kwargs.pop("subfolder", "")

    #     from_pipeline = kwargs.pop("_from_pipeline", None)
    #     from_auto_class = kwargs.pop("_from_auto", False) #自动false


    #     user_agent = {"file_type": "image processor", "from_auto_class": from_auto_class} #
    #     if from_pipeline is not None:
    #         user_agent["using_pipeline"] = from_pipeline

    #     if is_offline_mode() and not local_files_only:
    #         logger.info("Offline mode: forcing local_files_only=True")
    #         local_files_only = True

    #     model_name_or_path = str(model_name_or_path)
    #     is_local = os.path.isdir(model_name_or_path)
    #     if os.path.isdir(model_name_or_path):
    #         image_processor_file = os.path.join(model_name_or_path, IMAGE_PROCESSOR_NAME)
    #     if os.path.isfile(model_name_or_path):
    #         resolved_image_processor_file = model_name_or_path
    #         is_local = True
    #     elif is_remote_url(model_name_or_path):
    #         image_processor_file = model_name_or_path
    #         resolved_image_processor_file = download_url(model_name_or_path)
    #     else:
    #         image_processor_file = IMAGE_PROCESSOR_NAME
    #         try:
    #             # Load from local folder or from cache or download from model Hub and cache
    #             resolved_image_processor_file = cached_file( #本地缓存加载或者hub缓存加载
    #                 model_name_or_path,
    #                 image_processor_file,  #'preprocessor_config.json'
    #                 cache_dir=cache_dir, #none
    #                 force_download=force_download, #none
    #                 proxies=proxies, #false
    #                 resume_download=resume_download, #none
    #                 local_files_only=local_files_only, #false
    #                 token=token, #none
    #                 user_agent=user_agent,
    #                 revision=revision,
    #                 subfolder=subfolder,
    #             )
    #         except EnvironmentError:
    #             # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
    #             # the original exception.
    #             raise
    #         except Exception:
    #             # For any other exception, we throw a generic error.
    #             raise EnvironmentError(
    #                 f"Can't load image processor for '{model_name_or_path}'. If you were trying to load"
    #                 " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
    #                 f" same name. Otherwise, make sure '{model_name_or_path}' is the correct path to a"
    #                 f" directory containing a {IMAGE_PROCESSOR_NAME} file"
    #             )

    #     try:
    #         # Load image_processor dict
    #         with open(resolved_image_processor_file, "r", encoding="utf-8") as reader:
    #             text = reader.read()
    #         image_processor_dict = json.loads(text)

    #     except json.JSONDecodeError:
    #         raise EnvironmentError(
    #             f"It looks like the config file at '{resolved_image_processor_file}' is not a valid JSON file."
    #         )

    #     if is_local:
    #         logger.info(f"loading configuration file {resolved_image_processor_file}")
    #     else:
    #         logger.info(
    #             f"loading configuration file {image_processor_file} from cache at {resolved_image_processor_file}"
    #         )

    #     if not is_local:
    #         if "auto_map" in image_processor_dict:
    #             image_processor_dict["auto_map"] = add_model_info_to_auto_map(
    #                 image_processor_dict["auto_map"], model_name_or_path
    #             )
    #         if "custom_pipelines" in image_processor_dict:
    #             image_processor_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
    #                 image_processor_dict["custom_pipelines"], model_name_or_path
    #             )
    #     return image_processor_dict, kwargs



    @classmethod
    def get_image_processor_dict(
        cls, model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        image processor of type [`~image_processor_utils.ImageProcessingMixin`] using `from_dict`.

        Parameters:
            model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the image processor object.
        """
        # Step 1: Extract arguments from kwargs
        cache_dir, force_download, resume_download, proxies, token, local_files_only, revision, subfolder, user_agent = cls._extract_kwargs(kwargs)

        # Step 2: Handle offline mode
        local_files_only = cls._handle_offline_mode(local_files_only)

        # Step 3: Resolve the image processor file path
        resolved_image_processor_file, is_local = cls._resolve_image_processor_file(
            model_name_or_path, cache_dir, force_download, proxies, resume_download, local_files_only, token, user_agent, revision, subfolder
        )

        # Step 4: Load the image processor dictionary
        image_processor_dict = cls._load_image_processor_dict(resolved_image_processor_file, is_local)

        # Step 5: Add extra information for remote files
        cls._add_remote_info(image_processor_dict, model_name_or_path, is_local)

        return image_processor_dict, kwargs

    # Helper functions
    @staticmethod
    def _extract_kwargs(kwargs):
        """Extract and initialize required arguments from kwargs."""
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        user_agent = {"file_type": "image processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline
        return (
            cache_dir,
            force_download,
            resume_download,
            proxies,
            token,
            local_files_only,
            revision,
            subfolder,
            user_agent,
        )

    @staticmethod
    def _handle_offline_mode(local_files_only):
        """Handle offline mode by enforcing local file usage."""
        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True
        return local_files_only

    @staticmethod
    def _resolve_image_processor_file(
        model_name_or_path, cache_dir, force_download, proxies, resume_download, local_files_only, token, user_agent, revision, subfolder
    ):
        """Resolve the image processor file path based on input model_name_or_path."""
        model_name_or_path = str(model_name_or_path)
        is_local = os.path.isdir(model_name_or_path)

        if os.path.isdir(model_name_or_path):
            image_processor_file = os.path.join(model_name_or_path, IMAGE_PROCESSOR_NAME)
            resolved_image_processor_file = image_processor_file
        elif os.path.isfile(model_name_or_path):
            resolved_image_processor_file = model_name_or_path
            is_local = True
        elif is_remote_url(model_name_or_path):
            image_processor_file = model_name_or_path
            resolved_image_processor_file = download_url(model_name_or_path)
        else:
            image_processor_file = IMAGE_PROCESSOR_NAME
            try:
                resolved_image_processor_file = cached_file(
                    model_name_or_path,
                    image_processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                )
            except EnvironmentError as e:
                raise e
            except Exception:
                raise EnvironmentError(
                    f"Can't load image processor for '{model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{model_name_or_path}' is the correct path to a"
                    f" directory containing a {IMAGE_PROCESSOR_NAME} file"
                )
        return resolved_image_processor_file, is_local

    @staticmethod
    def _load_image_processor_dict(resolved_image_processor_file, is_local):
        """Load the image processor dictionary from the resolved file."""
        try:
            with open(resolved_image_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            image_processor_dict = json.loads(text)
        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_image_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_image_processor_file}")
        else:
            logger.info(
                f"loading configuration file from cache at {resolved_image_processor_file}"
            )
        return image_processor_dict

    @staticmethod
    def _add_remote_info(image_processor_dict, model_name_or_path, is_local):
        """Add extra information to the image processor dictionary if it's loaded from remote."""
        if not is_local:
            if "auto_map" in image_processor_dict:
                image_processor_dict["auto_map"] = add_model_info_to_auto_map(
                    image_processor_dict["auto_map"], model_name_or_path
                )
            if "custom_pipelines" in image_processor_dict:
                image_processor_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                    image_processor_dict["custom_pipelines"], model_name_or_path
                )




    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~image_processing_utils.ImageProcessingMixin`] from a Python dictionary of parameters.

        Args:
            image_processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the image processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~image_processing_utils.ImageProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the image processor object.

        Returns:
            [`~image_processing_utils.ImageProcessingMixin`]: The image processor object instantiated from those
            parameters.
        """
        image_processor_dict = image_processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # The `size` parameter is a dict and was previously an int or tuple in feature extractors.
        # We set `size` here directly to the `image_processor_dict` so that it is converted to the appropriate
        # dict within the image processor and isn't overwritten if `size` is passed in as a kwarg.
        if "size" in kwargs and "size" in image_processor_dict:
            image_processor_dict["size"] = kwargs.pop("size")
        if "crop_size" in kwargs and "crop_size" in image_processor_dict:
            image_processor_dict["crop_size"] = kwargs.pop("crop_size")

        image_processor = cls(**image_processor_dict)

        # Update image_processor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(image_processor, key):
                setattr(image_processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Image processor {image_processor}")
        if return_unused_kwargs:
            return image_processor, kwargs
        else:
            return image_processor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this image processor instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["image_processor_type"] = self.__class__.__name__

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates a image processor of type [`~image_processing_utils.ImageProcessingMixin`] from the path to a JSON
        file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A image processor of type [`~image_processing_utils.ImageProcessingMixin`]: The image_processor object
            instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        image_processor_dict = json.loads(text)
        return cls(**image_processor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this image_processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoImageProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom image processors as the ones
        in the library are already mapped with `AutoImageProcessor `.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoImageProcessor "`):
                The auto class to register this new image processor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def fetch_images(self, image_url_or_urls: Union[str, List[str]]):
        """
        Convert a single or a list of urls into the corresponding `PIL.Image` objects.

        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
                " Safari/537.36"
            )
        }
        if isinstance(image_url_or_urls, list):
            return [self.fetch_images(x) for x in image_url_or_urls]
        elif isinstance(image_url_or_urls, str):
            response = requests.get(image_url_or_urls, stream=True, headers=headers)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        else:
            raise TypeError(f"only a single or a list of entries is supported but got type={type(image_url_or_urls)}")


ImageProcessingMixin.push_to_hub = copy_func(ImageProcessingMixin.push_to_hub)
if ImageProcessingMixin.push_to_hub.__doc__ is not None:
    ImageProcessingMixin.push_to_hub.__doc__ = ImageProcessingMixin.push_to_hub.__doc__.format(
        object="image processor", object_class="AutoImageProcessor", object_files="image processor file"
    )
