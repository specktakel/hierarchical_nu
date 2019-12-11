"""
Module for collecting all-purpose baseclasses
"""

__all__ = ["NamedObject"]


class NamedObject(object):
    """Class implementing a named object"""

    @property
    def name(self):
        return self._name
