import ee
from geetiles import utils
import rasterio
import numpy as np

class DatasetDefinition:

    """ 
    the gee collection applies the transformation 10 log_10 \sigma to the backscattering.
    this results in a range of values of approx [-30,0]

    this class transforms that range into [0,255] so that it can be stored as uint8 without
    loosing too much precision, and saving storage space.
    """

    def __init__(self, dataset_name):
        self.year = dataset_name.split("-")[-1]
        self.dataset_name = dataset_name
        try:
            year = int(self.year)
        except Exception as e:
            raise ValueError(f"could not find year in {dataset_name}")

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_gee_image(self, **kwargs):
        s1grd = None
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

        for month, dates in months.items():

            sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
                        .filterDate(dates[0], dates[1])

            vvasc = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
                        .select(['VV'])\
                        .count()\
                        .rename(f'{month}_vvasc')

            vvdes = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))\
                        .select(['VV'])\
                        .count()\
                        .rename(f'{month}_vvdes')

            vhasc = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
                        .select(['VH'])\
                        .count()\
                        .rename(f'{month}_vhasc')

            vhdes = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))\
                        .select(['VH'])\
                        .count()\
                        .rename(f'{month}_vhdes')

            if s1grd is None:
                s1grd = vvasc
            else:
                s1grd = s1grd.addBands(vvasc)
            
            s1grd = s1grd.addBands(vvdes)\
                         .addBands(vhasc)\
                         .addBands(vhdes)

            
        return s1grd

    def map_values(self, array):
        return array

    def get_dtype(self):
        return 'uint8'
