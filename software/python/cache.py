"""
Caching module

Implements a simple class that stores intermediate data, such as created
when creating parametrizations for the detector models
"""
from typing import Union
from contextlib import contextmanager
import os


class CacheDirNotSetException(Exception):
    pass


class _MCache(type):
    """
    Metaclass for Cache

    used to overload Cache.__contains__
    """

    def __contains__(self, item):
        if self._cache_dir is None:
            raise CacheDirNotSetException()

        return os.path.exists(os.path.join(self._cache_dir, item))


class Cache(metaclass=_MCache):
    """
    Simple cache

    Wraps `open()` to append the cache directory name
    """

    _cache_dir: Union[None, str] = None

    @classmethod
    def set_cache_dir(cls, cache_dir: str):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cls._cache_dir = cache_dir

    @classmethod
    @contextmanager
    def open(cls, filename: str, modifier: str = "r"):
        if cls._cache_dir is None:
            raise CacheDirNotSetException()

        with open(os.path.join(cls._cache_dir, filename), modifier) as fr:
            yield fr

    @classmethod
    def clear_cache(cls, dry_run=True):
        if cls._cache_dir is None:
            raise CacheDirNotSetException()

        for f in os.listdir(cls._cache_dir):
            if dry_run:
                print("deleting", f)
            else:
                os.unlink(os.path.join(cls._cache_dir, f))


if __name__ == "__main__":
    Cache.set_cache_dir(".cache")

    print("test" in Cache)

    with Cache.open("test", "w") as fr:
        pass

    print("test" in Cache)

    Cache.clear_cache(dry_run=False)
