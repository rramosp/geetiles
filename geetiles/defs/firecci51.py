import ee
from geetiles import utils

# this class is here only for legacy as it got 
# renamed to esaworldcover2020

class DatasetDefinition:

    def __init__(self, dataset_def):
        self.dataset_def = dataset_def

    def get_dataset_name(self):
        return self.dataset_def
    
    def get_gee_image(self):

        cciburned = None
        year = '2020'

        for year in ['2016', '2017', '2018', '2019', '2020']:

            burnedyear = ee.ImageCollection('ESA/CCI/FireCCI/5_1')\
                        .filterDate(f'{year}-01-01', f'{year}-12-31')\
                        .select(['BurnDate', 'ConfidenceLevel', 'LandCover'])\
                        .max()\
                        .rename([f'BurnDate_{year}', f'ConfidenceLevel_{year}', f'LandCover_{year}'])

            
            if cciburned is None:
                cciburned = burnedyear
            else:
                cciburned = cciburned.addBands(burnedyear)

        return cciburned

    def map_values(self, array):
        return array
    
    def get_dtype(self):
        return 'int16'