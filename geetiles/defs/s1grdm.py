import ee
from geetiles import utils

class DatasetDefinition:

    def __init__(self, dataset_name):

        if not len(dataset_name.split("-")) == 3:
            raise ValueError(f"invalid name '{dataset_name}' must be of the form, for instance 's1grdm-2020-des'")

        self.year = dataset_name.split("-")[-2]
        self.direction = dataset_name.split("-")[-1]
        self.dataset_name = dataset_name

        if not self.direction in ['asc', 'desc']:
            raise ValueError(f"invalid direction '{self.direction}', must be 'asc' or 'dec'")

        try:
            year = int(self.year)
        except Exception as e:
            raise ValueError(f"could not find year in {dataset_name}")

    def get_dataset_name(self):
        return self.dataset_name
    
    def get_gee_image(self, **kwargs):
        s1grd = None
        year = self.year
        seasons = {
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
        
        dirstr = 'ASCENDING' if self.direction == 'asc' else 'DESCENDING'

        for month, dates in seasons.items():

            sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
                        .filterDate(dates[0], dates[1])

            vv = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', dirstr))\
                        .select(['VV'])\
                        .mean()\
                        .rename(f'{month}_vv')

            vh = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                        .filter(ee.Filter.eq('orbitProperties_pass', dirstr))\
                        .select(['VH'])\
                        .mean()\
                        .rename(f'{month}_vh')

            if s1grd is None:
                s1grd = vv
            else:
                s1grd = s1grd.addBands(vv)
            
            s1grd = s1grd.addBands(vh)

            
        return s1grd

    def map_values(self, array):
        return array
    
    def get_dtype(self):
        return 'uint16'
