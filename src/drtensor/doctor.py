from typing import Dict

from torch import nn

from .figures import ModuleFigure
from .hooks import RecordingHandler


class DrTensor:
    def __init__(
        self,
        **modules,
    ):
        self.module_objects: Dict[str, nn.Module] = dict()
        self.module_map: Dict[str, ModuleFigure] = dict()
        for name, module in modules.items():
            for subname, module in module.named_modules():
                module_key = f"{name}.{subname}" if subname != "" else name
                self.module_objects[module_key] = module
                self.module_map[module_key] = ModuleFigure.from_module(
                    name=module_key,
                    module=module,
                )

        # set up the parent-child relationship
        for name, module_figure in self.module_map.items():
            if module_figure.hierachy > 1:
                parent_name = ".".join(name.split(".")[:-1])
                if parent_name not in self.module_map:
                    continue
                self.module_map[parent_name].children.append(module_figure)
                module_figure.parent = self.module_map[parent_name]
        self.log = []

    def __enter__(self):
        self.hook_handles = dict()
        for name, module_figure in self.module_map.items():
            hook = RecordingHandler(
                module_figure=module_figure,
                log=self.log,
                module_map=self.module_map,
            )
            handle = module_figure.module.register_forward_hook(hook, with_kwargs=True)
            self.hook_handles.update({name: handle})
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name, handle in self.hook_handles.items():
            handle.remove()
