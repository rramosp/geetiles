import ee
from geetiles import utils

class DatasetDefinition:

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
        seasons = {'winter': [f'{int(year)-1:4d}-12-01', f'{year}-02-28'],
                'spring': [f'{year}-03-01', f'{year}-05-31'],
                'summer': [f'{year}-06-01', f'{year}-08-31'],
                'fall':   [f'{year}-09-01', f'{year}-11-30'],
                }

        for season, dates in seasons.items():

            sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
                        .filterDate(dates[0], dates[1])

            vvasc = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
                        .select(['VV'])\
                        .mean()\
                        .rename(f'{season}_vvasc')


            vvdes = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))\
                        .select(['VV'])\
                        .mean()\
                        .rename(f'{season}_vvdes')

            vhasc = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))\
                        .select(['VH'])\
                        .mean()\
                        .rename(f'{season}_vhasc')


            vhdes = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))\
                        .select(['VH'])\
                        .mean()\
                        .rename(f'{season}_vhdes')


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
        return 'uint16'
