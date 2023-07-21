"""Utilities for converting and operating on ONNX, JIT and torch types."""
from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch
from torch._subclasses import fake_tensor

if TYPE_CHECKING:
    import onnx.defs.OpSchema.AttrType  # type: ignore[import]


# Enable both TorchScriptTensor and torch.Tensor to be tested
# for dtype in OpSchemaWrapper.
@runtime_checkable
class TensorLike(Protocol):
    @property
    def dtype(self) -> Optional[torch.dtype]:
        ...


def is_torch_complex_dtype(tensor: TensorLike) -> bool:
    # NOTE: This is needed as TorchScriptTensor is nor supported by torch.is_complex()
    return tensor.dtype in _COMPLEX_TO_FLOAT


def from_complex_to_float(dtype: torch.dtype) -> torch.dtype:
    return _COMPLEX_TO_FLOAT[dtype]


def from_sym_value_to_torch_dtype(sym_value: SYM_VALUE_TYPE) -> torch.dtype:
    return _SYM_TYPE_TO_TORCH_DTYPE[type(sym_value)]


def from_torch_dtype_to_onnx_dtype_str(dtype: Union[torch.dtype, type]) -> Set[str]:
    return _TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[dtype]


def from_python_type_to_onnx_attribute_type(
    dtype: type, is_sequence: bool = False
) -> Optional[onnx.defs.OpSchema.AttrType]:
    import onnx.defs  # type: ignore[import]

    _PYTHON_TYPE_TO_ONNX_ATTRIBUTE_TYPE = {
        float: onnx.defs.OpSchema.AttrType.FLOAT,
        int: onnx.defs.OpSchema.AttrType.INT,
        str: onnx.defs.OpSchema.AttrType.STRING,
        bool: onnx.defs.OpSchema.AttrType.INT,
    }

    _SEQUENCE_TYPE_TO_ONNX_ATTRIBUTE_TYPE = {
        float: onnx.defs.OpSchema.AttrType.FLOATS,
        int: onnx.defs.OpSchema.AttrType.INTS,
        str: onnx.defs.OpSchema.AttrType.STRINGS,
        bool: onnx.defs.OpSchema.AttrType.INTS,
    }

    if is_sequence:
        return _SEQUENCE_TYPE_TO_ONNX_ATTRIBUTE_TYPE.get(dtype)
    return _PYTHON_TYPE_TO_ONNX_ATTRIBUTE_TYPE.get(dtype)


# NOTE: this is a mapping from torch dtype to a set of compatible onnx types
# It's used in dispatcher to find the best match overload for the input dtypes
_TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS: Dict[
    Union[torch.dtype, type], Set[str]
] = {
    torch.bfloat16: {"tensor(bfloat16)"},
    torch.bool: {"tensor(bool)"},
    torch.float64: {"tensor(double)"},
    torch.float32: {"tensor(float)"},
    torch.float16: {"tensor(float16)"},
    torch.int16: {"tensor(int16)"},
    torch.int32: {"tensor(int32)"},
    torch.int64: {"tensor(int64)"},
    torch.int8: {"tensor(int8)"},
    torch.uint8: {"tensor(uint8)"},
    str: {"tensor(string)"},
    int: {"tensor(int16)", "tensor(int32)", "tensor(int64)"},
    float: {"tensor(float16)", "tensor(float)", "tensor(double)"},
    bool: {"tensor(int32)", "tensor(int64)", "tensor(bool)"},
}

_COMPLEX_TO_FLOAT: Dict[torch.dtype, torch.dtype] = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,  # NOTE: ORT doesn't support torch.float64
}
_SYM_TYPE_TO_TORCH_DTYPE = {
    torch.SymInt: torch.int64,
    torch.SymFloat: torch.float32,
    torch.SymBool: torch.bool,
}

SYM_VALUE_TYPE = Union[torch.SymInt, torch.SymFloat, torch.SymBool]
META_VALUE_TYPE = Union[fake_tensor.FakeTensor, SYM_VALUE_TYPE]
# NOTE: Belows are from torch/fx/node.py
BaseArgumentTypes = Union[
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.Tensor,
    torch.device,
    torch.memory_format,
    torch.layout,
    torch._ops.OpOverload,
]
Argument = Optional[
    Union[
        Tuple[Any, ...],  # actually Argument, but mypy can't represent recursive types
        List[Any],  # actually Argument
        Dict[str, Any],  # actually Argument
        slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
        range,
        "torch.fx.Node",
        BaseArgumentTypes,
    ]
]
