import ee
from geetiles import utils
import rasterio
import os
from .. import utils

# this class is here only for legacy as it got 
# renamed to esaworldcover2020

class DatasetDefinition:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_dataset_name(self):
        return self.dataset_name
    
    def must_get_gee_image(self, filename):
        if os.path.isfile(filename) or \
           os.path.isfile(f"{filename}.nodata"):
            return False
        else:
            return True
    
    def get_gee_image(self, **kwargs):

        cciburned = None
        year = '2020'

        for year in ['2016', '2017', '2018', '2019', '2020']:

            burnedyear = ee.ImageCollection('ESA/CCI/FireCCI/5_1')\
                        .filterDate(f'{year}-01-01', f'{year}-12-31')\
                        .select(['BurnDate', 'ConfidenceLevel', 'LandCover'])\
                        .max()\
                        .rename([f'{year}_BurnDate', f'{year}_ConfidenceLevel', f'{year}_LandCover'])
            
            if cciburned is None:
                cciburned = burnedyear
            else:
                cciburned = cciburned.addBands(burnedyear)

        return cciburned

    def post_process_tilefile(self, filename):
        # open raster and check if there are any burnedout area
        # if not, remove it 
        with rasterio.open(filename) as src:
            x = src.read()

        if sum([x[i].sum() for i in range(x.shape[0])][::3])==0:
            os.remove(filename)
            utils.touch(f"{filename}.nodata")

    def map_values(self, array):
        return array
    
    def get_dtype(self):
        return 'int16'