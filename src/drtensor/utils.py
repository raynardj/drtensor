from typing import Callable


def apply_tree(callback: Callable):
    """
    Traverse the tree and call the callback function on elements

    And keep the tree structure
    """

    def apply_tree_fn(tree):
        if isinstance(tree, dict):
            return {k: apply_tree_fn(v) for k, v in tree.items()}
        elif isinstance(tree, list):
            return [apply_tree_fn(v) for v in tree]
        elif isinstance(tree, tuple):
            return tuple(apply_tree_fn(v) for v in tree)
        else:
            return callback(tree)

    return apply_tree_fn
