"""
Dataset generator for bigclonebench dataset

Provides streaming access to java files
"""

from pathlib import Path
import os

DEFAULT_BIGCLONEBENCH_ROOT = Path(
    os.getenv("BIGCLONEBENCH_ROOT", Path.home() / "datasets" / "bigclonebench")
)


def _read_java(java_path: Path) -> str:
    """
    Reads a single java file from path
    """
    return java_path.read_text(encoding="utf-8", errors="ignore")


def samples_generator(dataset_root: str | Path = DEFAULT_BIGCLONEBENCH_ROOT):
    """
    Generator for retrieving file paths from BigCloneBench samples folder.

    @param dataset_root: Path to BigCloneBench dataset root
    @return Generator of Path objects for each .java file in samples/
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    samples_path = dataset_root / "samples"

    if not samples_path.exists():
        raise ValueError(f"samples folder not found at {samples_path}")

    for java_file in samples_path.glob('*.java'):
        yield _read_java(java_file)


def default_generator(dataset_root: str | Path = DEFAULT_BIGCLONEBENCH_ROOT):
    """
    Generator for retrieving file paths from BigCloneBench default folder.

    @param dataset_root: Path to BigCloneBench dataset root
    @return Generator of Path objects for each .java file in default/
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    default_path = dataset_root / "default"

    if not default_path.exists():
        raise ValueError(f"default folder not found at {default_path}")

    for java_file in default_path.glob('*.java'):
        yield _read_java(java_file)
