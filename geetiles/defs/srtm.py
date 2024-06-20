import ee
from geetiles import utils

# this class is here only for legacy as it got 
# renamed to esaworldcover2020

class DatasetDefinition:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_gee_image(self, **kwargs):
        r = ee.Image('CGIAR/SRTM90_V4')
        slope = ee.Terrain.slope(r)
        r = r.addBands(slope)
        return r

    def map_values(self, array):
        return array
    
    def get_dtype(self):
        return 'uint16'
