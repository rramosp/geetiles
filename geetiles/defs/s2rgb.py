import ee
import rasterio
import os
import numpy as np
from skimage import exposure

class DatasetDefinition:

    def __init__(self, dataset_name):
        dataset_name_components = dataset_name.split("-")
        if len(dataset_name_components)!=2:
            raise ValueError("incorrect dataset name. must be 's2rgb-2020' or the year you want")
        
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

        gee_image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
                        .filterDate('2020-01-01', '2020-12-31')\
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))\
                        .map(maskS2clouds)\
                        .select('B4', 'B3', 'B2')\
                        .median()\
                        .visualize(min=0, max=0.3)
        
        return gee_image
    
    def post_process_tilefile(self, filename):

        # open raster again to adjust 
        with rasterio.open(filename) as src:
            x = src.read()
            x = np.transpose(x, [1,2,0])
            m = src.read_masks()
            profile = src.profile.copy()
            band_names = src.descriptions

        x = exposure.adjust_gamma(x, gamma=.8, gain=1.2)

        # write enhanced image
        with rasterio.open(filename, 'w', **profile) as dest:
            for i in range(src.count):
                dest.write(x[:,:,i], i+1)      
                dest.set_band_description(i+1, band_names[i])

    
    def get_dtype(self):
        return 'uint8'