import subprocess
import inspect
from os import path

"""
Find the software version or git hash
"""


# import arbitrary class defined in the package to get some directory inside the package
from .lifetime import LifeTime

# get dir of arbitrary class
cwd = path.abspath(path.dirname(inspect.getfile(LifeTime)))

try:
    process = subprocess.Popen(
        ["git", "rev-parse", "HEAD"],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=cwd,
    )
    git_hash = process.communicate()[0].strip().decode("ascii")
    if "fatal" in git_hash:
        # we are not using git but a pip installation
        raise ValueError
except ValueError:
    process = subprocess.Popen(
        ["pip", "show", "hierarchical_nu"],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=cwd,
    )
    git_hash = process.communicate()[0].strip().decode("ascii")
    git_hash = git_hash.split("\n")[1].lstrip("Version: ")
