import ee
from geetiles import utils

class DatasetDefinition:

    def get_gee_image(self):
        return ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")

    def get_dataset_name(self):
        return 'soilph'

    def map_values(self, array):
        # take only soil ph at the first level
        return utils.apply_range_map(array[:,:,0], [52,55])
                     