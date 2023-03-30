import ee

class DatasetDefinition:

    def get_dataset_name(self):
        return 'sentinel2-rgb-median-2020'
    
    def get_gee_image(self):
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