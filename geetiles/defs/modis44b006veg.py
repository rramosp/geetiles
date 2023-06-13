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
        modis = None
        for year in ['2016', '2017', '2018', '2019', '2020']:
            modisyear = ee.ImageCollection('MODIS/006/MOD44B') \
                            .filterDate(f'{year}-01-01', f'{year}-12-31')\
                            .select(['Percent_Tree_Cover'])\
                            .first()\
                            .rename(f"Percent_Tree_Cover_{year}")
            if modis is None:
                modis = modisyear
            else:
                modis = modis.addBands(modisyear)
        return modis

    def map_values(self, array):
        return array
    
    def get_dtype(self):
        return 'uint8'