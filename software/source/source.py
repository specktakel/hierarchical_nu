import numpy as np
from abc import ABC


DIFFUSE = 0
POINT = 1


class Source(ABC):
    """
    Abstract base class for sources.
    """


    @property
    def src_type(self):

        return self._src_type

    
    @src_type.setter
    def src_type(self, value):

        if value is not DIFFUSE and value is not POINT:

            raise ValueError(str(value) + ' is not a recognised source type')
        
        self._src_type = value

        
    @property
    def tag(self):

        return self._tag

    
    @tag.setter
    def tag(self, value):

        self._tag = value

"""    
    @property
    def flux_model(self):

        return self._flux_model

    
    @flux_model.setter
    def flux_model(self, value):

        if not isinstance(value, FluxModel):

            raise ValueError(str(value) + ' is not a recognised flux model')

        else:

            self._flux_model = value
"""


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
            

class PointSource(Source):

    
    def __init__(self, coord, redshift, tag=None):

        super().__init__()

        self.src_type = POINT
        
        self._coord = coord

        self._redshift = redshift

        self.tag = tag

        
    @property
    def coord(self):

        return self._coord

    
    @coord.setter
    def coord(self, value):

        self._coord = value

        
    @property
    def redshift(self):

        return self._redshift

    
    @redshift.setter
    def redshift(self, value):

        self._redshift = value


class DiffuseSource(Source):

    
    def __init__(self, redshift=None, tag=None):

        super().__init__()

        self.src_type = DIFFUSE
        
        self._redshift = redshift

        
    @property
    def redshift(self):

        return self._redshift

    
    @redshift.setter
    def redshift(self, value):

        self._redshift = value

        
        
class TestSourceList(SourceList):

    def __init__(self, filename, flux_model=None):
        """
        Simple source list from test file used in 
        development. Can be adapted to different 
        catalogs.

        :param filename: File to read from.
        :param flux_model: Option to specify flux model for all sources.
        """

        super().__init__()
        
        self._filename = filename

        self._flux_model = flux_model

        self._read_from_file()
        

    def _read_from_file(self):

        import h5py

        with h5py.File(self._filename, 'r') as f:

            redshift = f['output/redshift'][()]

            position = f['output/position'][()]
            
        unit_vector = position/np.linalg.norm(position)

        ra, dec = uv_to_icrs(unit_vector)

        for r, d, z in zip(ra, dec, redshift):

            source = PointSource((r, d), z)

            self.add(source)
        

    def select_below_redshift(self, zth):

        self._zth = zth
        
        self.sources = [s for s in self.sources if s.redshift <= zth]
                

def uv_to_icrs(unit_vector):
    """
    convert unit vector to ICRS coords (ra, dec)
    """

    if len(np.shape(unit_vector)) > 1:

      theta = np.arccos(unit_vector.T[2])

      phi = np.arctan(unit_vector.T[1] / unit_vector.T[0])

    else:

        theta = np.arccos(unit_vector[2])

        phi = np.arccos(unit_vector[1] / unit_vector[0])

    ra, dec = spherical_to_icrs(theta, phi)

    return ra, dec

        
def spherical_to_icrs(theta, phi):
    """
    convert spherical coordinates to ICRS
    ra, dec.
    """

    ra = phi

    dec = np.pi/2 - theta

    return ra, dec


def lists_to_tuple(list1, list2):

    return  [(list1[i], list2[i]) for i in range(0, len(list1))] 


    
