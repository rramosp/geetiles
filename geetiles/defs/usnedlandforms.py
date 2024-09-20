import ee
from geetiles import utils

class DatasetDefinition:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_gee_image(self, **kwargs):
        r = ee.Image('CSP/ERGo/1_0/US/landforms').select('constant')
        return r

    def map_values(self, array):
        return array
    
    def get_dtype(self):
        return 'uint8'
