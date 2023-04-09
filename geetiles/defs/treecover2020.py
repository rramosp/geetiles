import ee
from geetiles import utils

class DatasetDefinition:

    def get_gee_image(self):
        gee_image = ee.ImageCollection('MODIS/006/MOD44B')\
                    .filterDate('2020-01-01', '2020-12-31')\
                    .first()\
                    .select('Percent_Tree_Cover')\
                    .visualize(min=0, max=100)        
        
        return gee_image

    def get_dataset_name(self):
        return 'treecover2020'

    def map_values(self, array):
        # discretize percentage of tree cover
        return utils.apply_range_map(array, [20, 40, 60, 80])
                     
    def get_dtype(self):
        return 'uint8'