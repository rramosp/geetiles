import ee
from geetiles import utils

# this class is here only for legacy as it got 
# renamed to esaworldcover2020

class DatasetDefinition:

    def get_dataset_name(self):
        return 'esa-world-cover'
    
    def get_gee_image(self):
        return ee.ImageCollection("ESA/WorldCover/v100").first()

    def map_values(self, array):
        return utils.apply_value_map(array, {0:0, 10:1, 20:2, 30:3, 40:4, 50:5, 60:6, 70:7, 80:8, 90:9, 95:10, 100:11})

    def get_dtype(self):
        return 'uint8'