from typing import Iterable, Union
import numpy as np  # type: ignore

__all__ = ["TArrayOrNumericIterable"]

TArrayOrNumericIterable = Union[np.ndarray, Iterable[float]]
