import ee
import rasterio
import os
import numpy as np
from skimage import exposure

class DatasetDefinition:

    def __init__(self, dataset_name):
        dataset_name_components = dataset_name.split("-")
        if len(dataset_name_components)!=2:
            raise ValueError("incorrect dataset name. must be 's2rgbm-2020' or the year you want")
        
        self.year = dataset_name_components[1]
        self.dataset_name = dataset_name
        try:
            year = int(self.year)
        except Exception as e:
            raise ValueError(f"could not find year in {dataset_name}")
        
    def get_dataset_name(self):
        return self.dataset_name
    
    def get_gee_image(self, **kwargs):
    
        def maskS2clouds(image):
            qa = image.select('QA60')

            # Bits 10 and 11 are clouds and cirrus, respectively.
            cloudBitMask = 1 << 10
            cirrusBitMask = 1 << 11

            # Both flags should be set to zero, indicating clear conditions.
            mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

            return image.updateMask(mask).divide(10000)

        year = self.year
        months = {
                '01': [f'{year}-01-01', f'{year}-01-31'],
                '02': [f'{year}-02-01', f'{year}-02-28'],
                '03': [f'{year}-03-01', f'{year}-03-31'],
                '04': [f'{year}-04-01', f'{year}-04-30'],
                '05': [f'{year}-05-01', f'{year}-05-31'],
                '06': [f'{year}-06-01', f'{year}-06-30'],
                '07': [f'{year}-07-01', f'{year}-07-31'],
                '08': [f'{year}-08-01', f'{year}-08-31'],
                '09': [f'{year}-09-01', f'{year}-09-30'],
                '10': [f'{year}-10-01', f'{year}-10-31'],
                '11': [f'{year}-11-01', f'{year}-11-30'],
                '12': [f'{year}-12-01', f'{year}-12-31'],
                }
        
        s2rgb = None
        for month, dates in months.items():

            sentinel1 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
                        .filterDate(dates[0], dates[1])
        
            t = sentinel1\
                            .filterDate(dates[0],dates[1])\
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))\
                            .map(maskS2clouds)\
                            .select('B4', 'B3', 'B2')\
                            .median()\
                            .visualize(min=0, max=0.3)\
                            .rename([f'{month}_{b}' for b in ['red', 'green', 'blue']])
                            
            if s2rgb is None:
                s2rgb = t
            else:
                s2rgb = s2rgb.addBands(t)

        return s2rgb
        
    def get_dtype(self):
        return 'uint8'