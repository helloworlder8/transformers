from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_internlm2 import *
    from .modeling_internlm2 import *
    from .templates import *
    from .configuration_intern_vit import *
    from .modeling_intern_vit import *
    from .tokenization_internlm2_fast import *
    from .configuration_phi3 import *
    from .modeling_phi3 import *
    from .tokenization_internlm2 import *
    from .configuration_sa2va_chat import *
    from .modeling_sa2va_chat import *
    from .flash_attention import *
    from .sam2 import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__) #文件名 文件路径     模型在的内存空间
