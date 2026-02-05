"""
Dataset generator for CodeXGlue

Provides streaming access to Java files
"""
from datasets import load_dataset  # pylint: disable=no-name-in-module
from data.data_generators.schema import CodeSample


def bigclonebench_hf_generator(split: str):
    """
    Generator for BigCloneBench from CodeXGLUE (HuggingFace).

    Args:
        split: "train", "validation", or "test"

    Yields:
        CodeSample
    """
    ds = load_dataset(
        "google/code_x_glue_cc_clone_detection_big_clone_bench",
        split=split,
    )

    for sample in ds:
        hf_label = int(sample["label"])

        # Normalize:
        # HF: 0 = clone, 1 = non-clone
        # OUR: 1 = clone, 0 = non-clone
        label = 1 if hf_label == 0 else 0
        assert label in (0, 1)

        yield CodeSample(
            code_a=sample["func1"],
            code_b=sample["func2"],
            label=label,
            dataset="bigclonebench",
        )
