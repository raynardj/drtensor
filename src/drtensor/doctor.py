from typing import Any, Dict
import json

from torch import nn

from .figures import ModuleFigure, TensorFigure
from .hooks import RecordingHandler
from .utils import apply_tree


def build_tensor_save_tree(tensor_data_dict: Dict[str, Any], tensor_figure_dict: Dict[str, TensorFigure]):

    def save(tensor_figure):
        if tensor_figure.uuid not in tensor_figure_dict:
            tensor_figure_dict[tensor_figure.uuid] = tensor_figure
        if tensor_figure.uuid in tensor_data_dict:
            return tensor_data_dict[tensor_figure.uuid]
        else:
            tensor_data_dict[tensor_figure.uuid] = tensor_figure.to_dict()
            return tensor_data_dict[tensor_figure.uuid]

    return apply_tree(save)


def get_tensor_uuid(tensor_figure):
    return tensor_figure.uuid


def build_tree_get_tensor_figure(tensor_figure_dict: Dict[str, TensorFigure]):
    def get_tensor_figure(uuid):
        return tensor_figure_dict[uuid]

    return apply_tree(get_tensor_figure)


tree_tensor_uuid = apply_tree(get_tensor_uuid)


class DrTensor:
    def __init__(
        self,
        is_recostructed: bool = False,
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

    def to_dict(self):
        module_figure_dict = dict()  # not dumping
        module_data_dict = dict()  # dumping

        tensor_figure_dict = dict()  # not dumping
        tensor_data_dict = dict()  # dumping

        module_to_parent = dict()
        module_to_children = dict()

        tensor_encounters = dict()
        module_encounters = dict()

        tree_save = build_tensor_save_tree(tensor_data_dict, tensor_figure_dict)
        for module_name, module_figure in self.module_map.items():
            module_figure_dict[module_figure.uuid] = module_figure
            module_data_dict[module_figure.uuid] = module_figure.to_dict()

            for encounter in module_figure.encounters:
                tree_save(encounter)

            if module_figure.parent is not None:
                module_to_parent[module_figure.uuid] = module_figure.parent.uuid

            if len(module_figure.children) > 0:
                module_to_children[module_child.uuid] = []
                for module_child in module_figure.children:
                    module_to_children[module_child.uuid].append(module_child.uuid)

        for tensor_uuid, tensor_figure in tensor_figure_dict.items():
            if len(tensor_figure.encounters) > 0:
                tensor_encounters[tensor_uuid] = []
                for encounter_dict in tensor_figure.encounters:
                    tensor_encounters[tensor_uuid].append(
                        {
                            "module_uuid": encounter_dict["module"].uuid,
                            "tensor_type": encounter_dict["tensor_type"],
                        }
                    )

        for module_uuid, module_figure in module_figure_dict.items():
            if len(module_figure.encounters) > 0:
                module_encounters[module_uuid] = []
                for encounter_dict in module_figure.encounters:
                    module_encounters[module_uuid].append(tree_tensor_uuid(encounter_dict))

        return dict(
            modules=module_data_dict,
            tensors=tensor_data_dict,
            module_to_parent=module_to_parent,
            module_to_children=module_to_children,
            tensor_encounters=tensor_encounters,
            module_encounters=module_encounters,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        obj = cls(is_recostructed=True)

        module_data_dict = data["modules"]
        tensor_data_dict = data["tensors"]

        module_figure_dict = dict()
        for uuid, module_data in module_data_dict.items():
            module_figure_dict[uuid] = ModuleFigure.from_dict(module_data)
        obj.module_map = module_figure_dict

        tensor_figure_dict = dict()
        for uuid, tensor_data in tensor_data_dict.items():
            tensor_figure_dict[uuid] = TensorFigure.from_dict(tensor_data)

        for module_uuid, parent_uuid in data["module_to_parent"].items():
            module_figure_dict[module_uuid].parent = module_figure_dict[parent_uuid]

        for module_uuid, children_uuids in data["module_to_children"].items():
            for child_uuid in children_uuids:
                module_figure_dict[module_uuid].children.append(module_figure_dict[child_uuid])

        for tensor_uuid, encounters in data["tensor_encounters"].items():
            tensor_figure = tensor_figure_dict[tensor_uuid]
            for encounter in encounters:
                module_figure = module_figure_dict[encounter["module_uuid"]]
                tensor_figure.encounters.append(
                    {
                        "module": module_figure,
                        "tensor_type": encounter["tensor_type"],
                    }
                )

        tree_get_tensor_figure = build_tree_get_tensor_figure(tensor_figure_dict)

        for module_uuid, encounters in data["module_encounters"].items():
            module_figure = module_figure_dict[module_uuid]
            for encounter in encounters:
                io_figure = tree_get_tensor_figure(encounter)
                module_figure.encounters.append(io_figure)

        return obj

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, json_file):
        with open(json_file, "w") as f:
            json.dump(self.to_dict(), f)

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
