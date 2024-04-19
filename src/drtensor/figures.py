from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import torch
from torch import nn

from .utils import tree_to_string


class TensorFigure:
    """
    A figure of a tensor
    It's the shell containing many required meta data of a tensor
    """

    def __init__(
        self,
        shape: Optional[List[int]],
        dtype: Optional[str],
        device: Optional[str],
        is_parameter: bool = False,
        uuid: Optional[str] = None,
    ):
        self.shape = list(shape) if shape is not None else None
        self.dtype: Optional[str] = str(dtype) if dtype is not None else None
        self.device: Optional[str] = str(device) if device is not None else None
        self.encounters: List[Dict[str, Any]] = []
        self.is_parameter = is_parameter
        if uuid is None:
            self.uuid: str = str(uuid4())
        else:
            self.uuid: str = uuid

    @property
    def shape_print(self):
        product = " x".join([str(dim) for dim in self.shape])
        return f"({product})"

    def __repr__(self):
        logo = "🍓" if self.is_parameter else "💎"
        return f"{logo} {self.shape_print}(💠{self.dtype}) @ 🍪{self.device}"

    def __len__(self) -> int:
        if self.shape is None:
            return 0
        else:
            return self.shape[0]

    @property
    def numel(self) -> int:
        """
        number of elements
        """
        if self.shape is None:
            return 0
        product = 1
        for dim in self.shape:
            product *= dim
        return product

    @classmethod
    def from_tensor(cls, tensor: Optional[Union[torch.Tensor, torch.nn.Parameter]]):
        if tensor is None:
            return cls(shape=None, dtype=None, device=None)
        return cls(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            is_parameter=isinstance(tensor, torch.nn.Parameter),
        )

    def to_dict(self):
        return dict(
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            is_parameter=self.is_parameter,
            uuid=self.uuid,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)


class ModuleFigure:
    """
    A figure of a module

    It's the shell containing many required meta data of a module

    Also the graph relation of the module
    """

    def __init__(
        self,
        name: str,
        class_name: str,
        annotations: Dict[str, Any],
        weights: Dict[str, TensorFigure],
        module: Optional[nn.Module] = None,
        uuid: Optional[str] = None,
    ):
        self.name: str = name
        self.class_name: str = class_name
        self.annotations = annotations
        self.weights = weights
        self.encounters: List[Dict[str, Any]] = []
        self.children: List[ModuleFigure] = []
        self.parent = None
        self.module: Optional[nn.Module] = module
        if uuid is None:
            self.uuid = str(uuid4())
        else:
            self.uuid = uuid

    def __repr__(self):
        return f"🎲 {self.name} ({self.class_name}), weights: {self.weights}"

    @property
    def hierachy(self) -> int:
        return len(self.name.split("."))

    @classmethod
    def from_module(cls, name, module):
        weights_map = dict()
        for weight_name, param in module.named_parameters(recurse=False):
            weights_map[weight_name] = TensorFigure.from_tensor(param)
        return cls(
            name=name,
            class_name=module.__class__.__name__,
            annotations=tree_to_string(module.forward.__annotations__),
            weights=weights_map,
            module=module,
        )

    def to_dict(self):
        return dict(
            name=self.name,
            class_name=self.class_name,
            annotations=self.annotations,
            weights={k: v.to_dict() for k, v in self.weights.items()},
            uuid=self.uuid,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            name=data["name"],
            class_name=data["class_name"],
            annotations=data["annotations"],
            weights={k: TensorFigure.from_dict(v) for k, v in data["weights"].items()},
            module=None,
            uuid=data["uuid"],
        )
