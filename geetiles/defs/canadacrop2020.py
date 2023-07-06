import ee
from geetiles import utils

class DatasetDefinition:

    def get_dataset_name(self):
        return 'canadacrop2020'
    
    def get_gee_image(self, **kwargs):
        return  ee.ImageCollection('AAFC/ACI')\
                  .filter(ee.Filter.date('2020-01-01', '2020-12-31'))\
                  .first()
        
    def map_values(self, array):
        return array

    def get_dtype(self):
        return 'uint8'