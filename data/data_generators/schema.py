"""
Schema for dataset output
"""
from dataclasses import dataclass


@dataclass
class CodeSample:
    """
    Represents a pair of code, an identifier label (plag or not plag) and the origin
    """
    code_a: str
    code_b: str | None
    label: int # 1 = Plag, 0 = Cool shit
    dataset: str
