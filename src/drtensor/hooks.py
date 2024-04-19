from typing import Any, Callable, Dict, List

from .figures import ModuleFigure, TensorFigure
from .utils import apply_tree


def build_tensor_save_tree(module_figure: ModuleFigure, tensor_type: str):
    def save_(tensor):
        tensor_figure = TensorFigure.from_tensor(tensor)
        tensor_figure.encounters.append(
            {
                "module": module_figure,
                "tensor_type": tensor_type,
            }
        )
        return tensor_figure

    tree_save = apply_tree(save_)
    tree_save.__name__ = f"save_{tensor_type}"
    return tree_save


# tree version of TensorFigure.from_tensor
# tree_build_tensor_figure: Callable = apply_tree(TensorFigure.from_tensor)


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
            args=build_tensor_save_tree(self.module_figure, "args")(input),
            output=build_tensor_save_tree(self.module_figure, "output")(output),
            kwargs=build_tensor_save_tree(self.module_figure, "kwargs")(kwargs),
        )
        self.module_figure.encounters.append(io_figure)

        self.log.append(
            dict(
                name=self.module_figure.name,
                module=self.module_figure,
                **io_figure,
            )
        )
