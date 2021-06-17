import json
import argparse


def args_update(original_parser, update_args):
    """

    Args:
        original_args: argparse.ArgumentParser().parse_args()
        update_args: str

    Returns:
        argparse.ArgumentParser().parse_args()
    """
    original_args = original_parser.parse_args()
    json_update_args = json.loads(update_args)
    json_original_args = vars(original_args)
    for key_arg, val_arg in json_update_args.items():
        json_update_args[key_arg] = type(json_original_args[key_arg])(json_update_args[key_arg].replace("-", "_"))

    temp_args = argparse.Namespace()
    temp_args.__dict__.update(json_update_args)
    args = original_parser.parse_args(namespace=temp_args)

    return args
