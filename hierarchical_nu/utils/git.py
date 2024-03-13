import subprocess
import inspect
from os import path

# import arbitrary class defined in the package to get some directory inside the package
from .lifetime import LifeTime

# get dir of arbitrary class
cwd = path.abspath(path.dirname(inspect.getfile(LifeTime)))


process = subprocess.Popen(
    ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE, cwd=cwd
)
git_hash = process.communicate()[0].strip().decode("ascii")
