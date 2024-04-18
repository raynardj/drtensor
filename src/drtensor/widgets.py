try:
    from ipywidgets import HTML, VBox, HBox, Button, Output
except ImportError:
    print("This module involves ipywidgets, which is not installed.")
    exit(1)

from typing import Callable

from scipy.__config__ import show
from .figures import ModuleFigure


def build_click(output: Output, child: ModuleFigure) -> Callable:
    """
    Create a click event for the button
    """

    def output_click(o):
        output.clear_output()
        with output:
            show_log_item_func: Callable = build_log_item_with_output(output)
            show_log_item_func(child)

    return output_click


def build_log_item_with_output(output: Output) -> Callable:
    def show_log_item(module_figure):
        module_name = module_figure.name
        weights_html = "".join([f"<li>{k}: {v}</li>" for k, v in module_figure.weights.items()])

        kwargs_html = ""
        if len(module_figure.encounters) > 0:
            kwargs = module_figure.encounters[-1]["kwargs"]
            if len(kwargs) > 0:
                kwargs_html = "<h4>Kwargs:</h4>" + str(kwargs)

        annotated_types_html = "".join([f"<li>{k}: {v.__name__}</li>" for k, v in module_figure.annotations.items()])

        args_html = ""
        if len(module_figure.encounters) > 0:
            args = module_figure.encounters[-1]["args"]
            if len(args) > 0:
                args_html = "".join([f"<li>{idx}: {v}</li>" for idx, v in enumerate(args)])

        output_html = ""
        if len(module_figure.encounters) > 0:
            output_html = module_figure.encounters[-1]["output"]
        html = HTML(
            f"""
            <h2>{module_name} ({module_figure.class_name})</h2> 
            <h3>Annotated Types:</h3>
            {annotated_types_html}
            <h3>Inputs:</h3>
            <h4>Arguments:</h4>
            {args_html}
            {kwargs_html}
            <h3>Output:</h3>
            {output_html}

            <h3>Weights:</h3>
            {weights_html}
        """
        )

        navigation_buttons = []
        if module_figure.parent is not None:
            parent_button = Button(description="⭐️ Parent")
            parent_button.on_click(build_click(output, module_figure.parent))
            navigation_buttons.append(parent_button)

        if len(module_figure.children) > 0:
            for child in module_figure.children:
                child_button = Button(description=f"🍼 {child.name}", layout=dict(width="500px", color="blue"))
                child_button.on_click(build_click(output, child))
                navigation_buttons.append(child_button)

        control = VBox(navigation_buttons)
        with output:
            display(VBox([html, control]))

    return show_log_item
