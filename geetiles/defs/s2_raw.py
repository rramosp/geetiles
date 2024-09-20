
import ee
import rasterio
import os
import numpy as np
from skimage import exposure

class DatasetDefinition:

    def __init__(self, dataset_name):
        dataset_name_components = dataset_name.split("-")
        if len(dataset_name_components)!=2:
            raise ValueError("incorrect dataset name. must be 's2-2020' or the year you want")
        
        self.year = dataset_name_components[1]
        self.dataset_name = dataset_name
        try:
            year = int(self.year)
        except Exception as e:
            raise ValueError(f"could not find year in {dataset_name}")
        
    def get_dataset_name(self):
        return self.dataset_name
    
    def get_gee_image(self, **kwargs):
    
        year = self.year
        seasons = {'winter': [f'{int(year)-1:4d}-12-01', f'{year}-02-28'],
                'spring': [f'{year}-03-01', f'{year}-05-31'],
                'summer': [f'{year}-06-01', f'{year}-08-31'],
                'fall':   [f'{year}-09-01', f'{year}-11-30'],
                }        

        band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

        chip = None
        for season, dates in seasons.items():

            geeimg = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
                            .filterDate(dates[0],dates[1])\
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))\
                            .select(band_names)\
                            .median()\
                            .rename([f'{season}_{b}' for b in band_names])
                            #.divide(10000).multiply(255).toByte()\
                            
            if chip is None:
                chip = geeimg 
            else:
                chip = chip.addBands(geeimg)

        return chip
    
    def get_dtype(self):
        return 'float32'