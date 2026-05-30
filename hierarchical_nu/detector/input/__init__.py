import inspect
from os import path

from ...events import Events

mceq = path.join(
    path.abspath(path.dirname(inspect.getfile(Events))), "detector", "input", "mceq"
)
