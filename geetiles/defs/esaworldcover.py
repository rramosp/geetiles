import ee
from geetiles import utils

class DatasetDefinition:

    def __init__(self, dataset_def='esaworldcover-2020'):

        self.dataset_def = dataset_def

        if dataset_def == 'esa-world-cover':
            # for legacy
            self.year = "2020"

        else:
            self.year = dataset_def.split("-")[-1]

    def get_dataset_name(self):
        return self.dataset_def
    
    def get_gee_image(self):
        if self.year == '2020':
            return ee.ImageCollection("ESA/WorldCover/v100").first()
        elif self.year == '2021':
            return ee.ImageCollection("ESA/WorldCover/v200").first()

        raise ValueError(f"invalid year {self.year} for esaworldcover")


    def map_values(self, array):
        return utils.apply_value_map(array, {0:0, 10:1, 20:2, 30:3, 40:4, 50:5, 60:6, 70:7, 80:8, 90:9, 95:10, 100:11})

    def get_dtype(self):
        return 'uint8'