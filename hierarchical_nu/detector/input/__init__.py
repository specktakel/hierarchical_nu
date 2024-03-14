import inspect
from os import path

from ..icecube import Refrigerator

mceq = path.join(
    path.abspath(path.dirname(inspect.getfile(Refrigerator))), "input", "mceq"
)
