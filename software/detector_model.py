"""
This module contains classes for modelling detectors
"""

from abc import ABCMeta, abstractmethod


class EffectiveArea(metaclass=ABCMeta):
	"""
	Implements baseclass for effective areas.

	Every effective area has to implement a call method,
	which returns the effective area as function of true energy and true
	zenith.
	"""

	@abstractmethod
	def __call__(self, **kwargs):
		pass

	@abstractmethod
	def setup(self):
		"""
		Download and or build all the required input data for calculating
		the effective area
		"""


class NortherTracksEffectiveArea(EffectiveArea):
	"""
	Effective area for the two-year Northern Tracks release:
	https://icecube.wisc.edu/science/data/HE_NuMu_diffuse

	"""

	def __call__(self, **kwargs):
		



class DetectorModel(metaclass=ABCMeta):


	@property
	def effective_area(self):
		return self._get_effective_area()

	@abstractmethod
	def _get_effective_area(self):
		return self.__get_effective_area
	

	@property
	def energy_resolution(self):
		return self._get_energy_resolution()

	@abstractmethod
	def _get_energy_resolution(self):
		return self._energy_resolution

	@property
	def angular_resolution(self):
		return _get_angular_resolution()

	@abstractmethod
	def _get_angular_resolution(self):
		self._angular_resolution






class Dataset(metaclass=ABCMeta):
	"""
	Baseline class for all datasets
	"""


