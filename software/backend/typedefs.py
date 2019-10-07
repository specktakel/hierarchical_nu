from typing import Iterable, Union,  Dict, Tuple, TYPE_CHECKING
import numpy as np  # type: ignore

if TYPE_CHECKING:
    from .stan_code import TListStrStanCodeBit
__all__ = ["TArrayOrNumericIterable", "TStanHookDict"]

TArrayOrNumericIterable = Union[np.ndarray, Iterable[float]]
TStanHookDict = Dict[str, Tuple[str, "TListStrStanCodeBit"]]
