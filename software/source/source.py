import numpy as np
from abc import ABC

from astropy.coordinates import SkyCoord

from astromodels import PointSource as PS
from astromodels import PointSource as ES
from astromodels.sources.source import Source, EXTENDED_SOURCE
from astromodels.core.spectral_component import SpectralComponent
from astromodels.core.tree import Node
from astromodels.core.units import get_units


class PointSource(PS):

    def __init__(self, source_name, redshift, **kwargs):
        """
        Override astromodels PointSource to add redshift.
        """

        super().__init__(source_name, **kwargs)

        self.redshift = redshift

    @property
    def redshift(self):

        return self._redshift


    @redshift.setter
    def redshift(self, value):

        if value < 0 or value > 10:

            raise ValueError(str(value) + ' is not a valid redshift.')

        else:

            self._redshift = value


class ExtendedSource(ES):

    def __init__(self, source_name, redshift, **kwargs):
        """
        Override astromodels ExtendedSource to add redshift.
        """

        super().__init__(source_name, **kwargs)

        
        self.redshift = redshift

        
    @property
    def redshift(self):

        return self._redshift


    @redshift.setter
    def redshift(self, value):

        if value < 0 or value > 10:

            raise ValueError(str(value) + ' is not a valid redshift.')

        else:

            self._redshift = value


            
class DiffuseSource(Source, Node):

    def __init__(self, source_name, redshift, spectral_shape=None, components=None):
        """
        Diffuse source for isotropic emission.
        """
        
        self.redshift = redshift
        
        assert (spectral_shape is not None) ^ (components is not None)

        # If the user specified only one component, make a list of one element with a default name ("main")

        if spectral_shape is not None:

            components = [SpectralComponent("main", spectral_shape)]

        Source.__init__(self, components, EXTENDED_SOURCE)

        # A source is also a Node in the tree

        Node.__init__(self, source_name)


        # Add a node called 'spectrum'
        
        spectrum_node = Node('spectrum')
        spectrum_node._add_children(self._components.values())

        self._add_child(spectrum_node)

        # Now set the units
        # Now sets the units of the parameters for the energy domain

        current_units = get_units()

        # Components in this case have energy as x and differential flux as y

        x_unit = current_units.energy
        y_unit = (current_units.energy * current_units.area * current_units.time) ** (-1)

        # Now set the units of the components
        for component in self._components.values():

            component.shape.set_units(x_unit, y_unit)

        
    @property
    def redshift(self):

        return self._redshift


    @redshift.setter
    def redshift(self, value):

        if value < 0 or value > 10:

            raise ValueError(str(value) + ' is not a valid redshift.')

        else:

            self._redshift = value


            
class SourceList(ABC):
    """
    Abstract base class for container of a list of sources.
    """

    def __init__(self):

        super().__init__()

        self.N = 0

        self._sources = []


    @property
    def sources(self):

        return self._sources
        

    @sources.setter
    def sources (self, value):

        if not isinstance(value, list):

            raise ValueError(str(value) + ' is not a list')

        if not isinstance(value[0], Source):
            
            raise ValueError(str(value) + ' is not a recognised source list')
        
        else:

            self._sources = value
            
            self.N = len(self._sources)

            
    def add(self, value):
        
        if not isinstance(value, Source):

            raise ValueError(str(value) + ' is not a recognised source')

        else:

            self._sources.append(value)

            self.N += 1

            
    def remove(self, index):           

            self._sources.pop(index)

            self.N -= 1

        
class TestSourceList(SourceList):

    def __init__(self, filename, spectral_shape=None):
        """
        Simple source list from test file used in 
        development. Can be adapted to different 
        catalogs.

        :param filename: File to read from.
        :param flux_model: Option to specify flux model for all sources.
        """

        super().__init__()
        
        self._filename = filename

        self._spectral_shape = spectral_shape

        self._read_from_file()


    @property
    def spectral_shape(self):

        return self._spectral_shape

    
    @spectral_shape.setter
    def spectral_shape(self, function):

        self._spectral_shape = function
        

    def _read_from_file(self):

        import h5py

        with h5py.File(self._filename, 'r') as f:

            redshift = f['output/redshift'][()]

            position = f['output/position'][()]
            
        unit_vector = position/np.linalg.norm(position)

        ra, dec = uv_to_icrs(unit_vector)

        for r, d, z in zip(np.rad2deg(ra), np.rad2deg(dec), redshift):

            source = PointSource('test_'+str(self.N), ra=r, dec=d, redshift=z,
                                 spectral_shape=self.spectral_shape)

            self.add(source)
        

    def select_below_redshift(self, zth):

        self._zth = zth
        
        self.sources = [s for s in self.sources if s.redshift <= zth]
                

def uv_to_icrs(unit_vector):
    """
    convert unit vector to ICRS coords (ra, dec)
    """

    if len(np.shape(unit_vector)) > 1:

        x = unit_vector.T[0]
        y = unit_vector.T[1]
        z = unit_vector.T[2]

    else:

        x = unit_vector[0]
        y = unit_vector[1]
        z = unit_vector[2]

    coord = SkyCoord(x, y, z, unit="mpc", representation_type="cartesian",
                     frame="icrs")
    coord.representation_type = "spherical"

    return coord.ra.rad, coord.dec.rad



    
