import ee
from geetiles import utils
import numpy as np

class DatasetDefinition:

    def get_gee_image(self):
        dataset = ee.ImageCollection('JRC/GHSL/P2016/POP_GPW_GLOBE_V1')\
                    .filter(ee.Filter.date('2015-01-01', '2015-12-31'))

        # since dataset returns an image collection with a single image
        # we use the median to get an image (other funcs such as 'first' wont work)
        populationCount = dataset.select('population_count').median()
        return populationCount

    def get_dataset_name(self):
        return 'humanpop2015'

    def map_values(self, array):
        return utils.apply_range_map(array, list(range(1,300,10)))

    def get_dtype(self):
        return 'uint16'

    def include_chip_in_dataset(self, chip_dict):
        #cprops = chip_dict['label_proportions']['partitions_aschip']
        #if '0' in cprops.keys() and cprops['0']==1. and np.random.random()>0.01:
        #    return False
        
        return True