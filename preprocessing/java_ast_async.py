import asyncio
from pathlib import Path
from typing import Iterator, Tuple

import javalang


def _parse_and_extract(java_path: Path):
    """
    Blocking function: parse Java file and extract methods.
    Runs in executor.
    """
    with java_path.open("r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    tree = javalang.parse.parse(source)

    results = []
    for _, node in tree.filter(
        (javalang.tree.MethodDeclaration,
         javalang.tree.ConstructorDeclaration)
    ):
        name = node.name if hasattr(node, "name") else "<constructor>"
        results.append((name, node))

    return results


async def extract_methods_async(java_path: str):
    """
    Async wrapper around javalang parsing.

    @param java_path: Path to a .java file
    @return: List of (method_name, ast_node)
    """
    path = Path(java_path)
    if not path.exists():
        raise FileNotFoundError(path)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _parse_and_extract, path)


def _parse_and_extract_classes(java_path: Path):
    """
    Blocking function: parse Java file and extract class-like declarations.
    Runs in executor.
    """
    with java_path.open("r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    tree = javalang.parse.parse(source)

    results = []

    for _, node in tree.filter((
        javalang.tree.ClassDeclaration,
        javalang.tree.InterfaceDeclaration,
        javalang.tree.EnumDeclaration,
    )):
        results.append((node.name, node))

    return results


async def extract_classes_async(java_path: str):
    """
    Async wrapper around javalang parsing for class-level declarations.

    @param java_path: Path to a .java file
    @return: List of (class_name, ast_node)
    """
    path = Path(java_path)
    if not path.exists():
        raise FileNotFoundError(path)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _parse_and_extract_classes, path)