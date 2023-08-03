import random
from os import path, walk
from typing import List

from PIL import Image


def find_files(dirpath: str, exts: List[str]):
    results = []
    for root, _, files in walk(dirpath):
        for file in files:
            for ext in exts:
                if file.lower().endswith(ext.lower()):
                    file = path.join(root, file)
                    results.append(file)
    return results


def find_fonts(dirpath: str):
    return find_files(dirpath, [".otf", ".ttf"])


def find_images(dirpath: str):
    exts = Image.registered_extensions()
    return find_files(dirpath, exts)


def rand_select(value):
    if isinstance(value, tuple):
        return random.uniform(*value)
    elif isinstance(value, list):
        return random.choice(value)
    else:
        return value
