from typing import Any, Callable, Dict, List

from .figures import ModuleFigure, TensorFigure
from .utils import apply_tree


# tree version of TensorFigure.from_tensor
tree_build_tensor_figure: Callable = apply_tree(TensorFigure.from_tensor)


class RecordingHandler:
    """
    A callable hook to be registered to a module
    """

    def __init__(
        self,
        module_figure: ModuleFigure,
        log: List[Dict[str, Any]],
        module_map: Dict[str, ModuleFigure],
    ):
        self.module_figure = module_figure
        self.module = module_figure.module
        self.module_map = module_map
        self.log = log

    def __call__(self, module, input, kwargs, output):
        io_figure = dict(
            args=tree_build_tensor_figure(input),
            output=tree_build_tensor_figure(output),
            kwargs=tree_build_tensor_figure(kwargs),
        )
        self.module_figure.encounters.append(io_figure)
        self.log.append(
            dict(
                name=self.module_figure.name,
                module=self.module_figure,
                **io_figure,
            )
        )
