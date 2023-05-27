import ee
from geetiles import utils

# this class is here only for legacy as it got 
# renamed to esaworldcover2020

class DatasetDefinition:

    def get_dataset_name(self):
        return 'modis44b006-veg2020'
    
    def get_gee_image(self):
        return ee.ImageCollection('MODIS/006/MOD44B') \
                 .filterDate('2020-01-01', '2020-12-31').first()
        
    def map_values(self, array):
        return array
    
    def get_dtype(self):
        return 'uint8'