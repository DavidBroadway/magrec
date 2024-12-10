import pathlib

import importlib
import os
import sys

__datapath__ = pathlib.Path(__file__).parent.resolve() / "data"
"""Standard path to the data directory."""

__logpath__ = pathlib.Path(__file__).parent.parent.resolve() / "logs"
"""Standard path to the log directory."""

def add_folder_to_path(folder):
    if folder not in sys.path:
        sys.path.append(folder)

def dynamic_import(module_name):
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        raise ImportError(f"Could not import module {module_name}: {e}")

def install_dynamic_import(folder):
    add_folder_to_path(folder)

    # Recursively walk through the data directory and import modules
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                relative_path = os.path.relpath(root, folder)
                module_name = os.path.splitext(file)[0]
                full_module_name = f"data.{relative_path.replace(os.sep, '.')}.{module_name}"
                full_module_name = full_module_name.replace('..', '.').strip('.')
                dynamic_import(full_module_name)