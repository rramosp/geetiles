
import itertools
import ee
import os
import shutil
import pickle
from shapely import wkt
import geopandas as gpd
import numpy as np
import shapely as sh
from joblib import delayed
from pyproj import CRS
from progressbar import progressbar as pbar
from skimage.io import imread
import pickle
from zipfile import ZipFile, ZIP_DEFLATED


from . import partitions
from . import utils

epsg4326 = utils.epsg4326


def split(tiles_file, 
          nbands, 
          angle,
          train_pct, 
          test_pct, 
          val_pct,
          foreign_tiles_name=None):
    
    p = partitions.PartitionSet.from_file(tiles_file)
    if foreign_tiles_name is None:
        p.split(nbands=nbands, angle=angle, 
                        train_pct=train_pct, test_pct=test_pct, val_pct=val_pct)
    else:
        p.split_per_partitions(nbands=nbands, angle=angle, 
                            train_pct=train_pct, test_pct=test_pct, val_pct=val_pct, 
                            other_partitions_id=foreign_tiles_name)
    p.save_splits()

def label_proportions_compute(tiles_file, 
                              labels_dataset_def):
    
    label_dataset_definition = utils.get_dataset_definition(labels_dataset_def)
    print ("loading tiles from", tiles_file, flush=True)
    p = partitions.PartitionSet.from_file(tiles_file)
    print(f"computing proportions for {len(p.data)} partitions")
    p.add_proportions(label_dataset_definition, n_jobs=1)
    print ("done!")

def label_proportions_from_foreign(tiles_file,
                                   foreign_tiles_file,
                                   labels_dataset_def):

    label_dataset_definition = utils.get_dataset_definition(labels_dataset_def)

    print ("loading primary tiles from", tiles_file, flush=True)
    p = partitions.PartitionSet.from_file(tiles_file)
    print ("loading foreign tiles from", foreign_tiles_file, flush=True)
    t = partitions.PartitionSet.from_file(foreign_tiles_file) 
    print ("intersecting geometries and computing label proportions from foreign tiles", flush=True)
    p.add_foreign_proportions(label_dataset_definition, t)    
    print ("done!")


def intersect_with_foreign(tiles_file, foreign_tiles_file):
    print ("loading primary tiles from", tiles_file, flush=True)
    p = partitions.PartitionSet.from_file(tiles_file)
    print ("loading foreign tiles from", foreign_tiles_file, flush=True)
    t = partitions.PartitionSet.from_file(foreign_tiles_file) 
    print ("intersecting geometries with foreign tiles", flush=True)
    p.add_foreign_partition(t)    
    print ("done!")


def download(tiles_file, 
             dataset_def, 
             pixels_lonlat, meters_per_pixel, 
             max_downloads,
             shuffle,
             skip_if_exists,
             ee_auth_mode,
             n_processes,
             skip_confirm=False):
    
    # sanity check
    if (pixels_lonlat is None and meters_per_pixel is None) or\
       (pixels_lonlat is not None and meters_per_pixel is not None):
        raise ValueError ("must specify exactly one of 'pixels_lonlat' or 'meters_per_pixel'")

    if pixels_lonlat is not None:
        try:
            pixels_lonlat = eval(pixels_lonlat)
            pixels_lonlat = [int(i) for i in pixels_lonlat]
            if not len(pixels_lonlat)==2:
                raise Exception
        except:
            raise ValueError("'pixels_lonlat' must be a tuple of two ints such as --pixels_lonlat [100,100]")


    dataset_definition = utils.get_dataset_definition(dataset_def)
    dtype = dataset_definition.get_dtype()

    print (f"""
using the following download specficication

tiles_file:        {tiles_file}
dataset_def        {dataset_def}
pixels_lonlat      {pixels_lonlat}
meters_per_pixel   {meters_per_pixel}
max_downloads      {max_downloads}
shuffle            {shuffle}
skip_if_exists     {skip_if_exists}
dtype              {dtype}
ee_auth_mode       {ee_auth_mode}
n_processes        {n_processes}

        """)
    
    if not skip_confirm:
        while True:
            yesno = input("confirm (y/N): ")        
            yesno = yesno.lower()
            if yesno.strip()=='':
                yesno = 'n'
            if yesno in ['y', 'n', 'yes', 'no']:
                break
        
        if yesno in ['n', 'no']:
            print ("abort!!")
            return
        
    # authenticate on google earth engine
    print ("authenticating to Google Earth Engine")
    if ee_auth_mode is None:
        try:
            print ("trying to use default gee credentials")
            ee.Authenticate(auth_mode = 'appdefault')    
        except:
            print ("could not authenticate with default gee credentials, using auth_method = 'notebook'")
            ee.Authenticate(auth_mode = 'notebook')
        
    else:
        ee.Authenticate(auth_mode = ee_auth_mode)

    ee.Initialize()

    gee_image = dataset_definition.get_gee_image()
    dataset_name = dataset_definition.get_dataset_name()

    print ("-----------------------------------------------")
    print (f"dataset name is '{dataset_name}'")
    print ("-----------------------------------------------")
    # download the tiles
    p = partitions.PartitionSet.from_file(tiles_file)

    # save gee_image_codestr
    dest_dir = p.get_downloaded_tiles_dest_dir(dataset_name)
    os.makedirs(dest_dir, exist_ok=True)
    with open(f"{dest_dir}.dataset_def.py", "w") as f:
        f.write(dataset_def)

    p.download_gee_tiles(gee_image, dataset_name, 
                         meters_per_pixel = meters_per_pixel, 
                         pixels_lonlat = pixels_lonlat,
                         dtype = dtype, 
                         shuffle = shuffle,
                         skip_if_exists = skip_if_exists,
                         enhance_images = None,
                         max_downloads=max_downloads)
    
    print("\ndone.")

def make_random_partitions(aoi_wkt_file, max_rectangle_size_meters, aoi_name, dest_dir, random_variance=0.1, ):
    """
    makes random partitions of the aoi
    """
    with open(aoi_wkt_file, "r") as f:
        aoi = wkt.loads(f.read()) 
        
    parts = partitions.PartitionSet(aoi_name, region=aoi)
    parts.reset_data().make_random_partitions(max_rectangle_size=max_rectangle_size_meters, random_variance=random_variance)
    print()
    parts.save_as(dest_dir, f'{max_rectangle_size_meters//1000}k')
    
    return parts.data

def make_grid(aoi_wkt_file, chip_size_meters, aoi_name, dest_dir):
    
    with open(aoi_wkt_file, "r") as f:
        aoi = wkt.loads(f.read()) 
            
    grid = build_grid(aoi=aoi, chip_size_meters=chip_size_meters)
    parts = partitions.PartitionSet(aoi_name, data=grid)
    print()
    parts.save_as(dest_dir, "aschips")
    return parts.data

def build_grid(aoi, chip_size_meters):
    """
    make a grid of squared tiles. The resulting tiles sides have
    constant lat and lon, as required by GEE, otherwise unaligned geometries 
    produce null pixels on the borders, when extracting the geometry from gee.

    aoi: a shapely object with the geometry to cover. must be in degrees (epsg4326)
    chip_size_meters: the length of each chip side in meters.

    returns: a GeoPandas dataframe in epsg4326
    """
    m = chip_size_meters
    
    # make a grid of points using utm crs
    aoi_utm = utils.get_utm_crs(*list(aoi.centroid.coords)[0])
    aoim = gpd.GeoDataFrame({'geometry': [aoi]}, crs=epsg4326).to_crs(aoi_utm).geometry[0]

    rcoords = np.r_[aoim.envelope.boundary.coords]
    minx, miny = rcoords.min(axis=0)
    maxx, maxy = rcoords.max(axis=0)
    rangex = maxx-minx
    rangey = maxy-miny
    gridx = int(rangex//m)
    gridy = int(rangey//m)
    
    def get_polygon(m, gx, gy, minx, miny):

        rlon, rlat = gx*m+minx, gy*m+miny
        point = sh.geometry.Point([rlon, rlat])

        # get point in lon/lat in degrees
        p4326 = gpd.GeoDataFrame({'geometry': [sh.geometry.Point([rlon, rlat])]},
                                          crs = aoi_utm).to_crs(epsg4326)
        clon,clat = list(p4326.geometry.values[0].coords)[0]

        # obtain how many meters per degree lon and lat in this region of the globe.
        # by doing the aritmethic in degrees (not in meters) we ensure tile sides have
        # constant lat and lon, as required by GEE, otherwise unaligned geometries 
        # produce null pixels on the borders.
        lon0,lat0 = list(gpd.GeoSeries([sh.geometry.Point([clon, clat])], crs=epsg4326).to_crs(aoi_utm).values[0].coords)[0]
        lon1,lat1 = list(gpd.GeoSeries([sh.geometry.Point([clon+0.001, clat])], crs=epsg4326).to_crs(aoi_utm).values[0].coords)[0]
        lon2,lat2 = list(gpd.GeoSeries([sh.geometry.Point([clon, clat+0.001])], crs=epsg4326).to_crs(aoi_utm).values[0].coords)[0]

        meters_per_degree_lon = (lon1-lon0) * 1000
        meters_per_degree_lat = (lat2-lat0) * 1000
        delta_degrees_lon =  ((m-1)/2) / meters_per_degree_lon
        delta_degrees_lat =  ((m-1)/2) / meters_per_degree_lat

        part =  sh.geometry.Polygon([[clon-delta_degrees_lon, clat-delta_degrees_lat], 
                                     [clon-delta_degrees_lon, clat+delta_degrees_lat],
                                     [clon+delta_degrees_lon, clat+delta_degrees_lat],
                                     [clon+delta_degrees_lon, clat-delta_degrees_lat],
                                     [clon-delta_degrees_lon, clat-delta_degrees_lat]])    
        return part    
    
    
    # create a polygon at each point
    print (f"inspecting {gridx*gridy} chips", flush=True)

    parts = utils.mParallel(n_jobs=-1, verbose=30)(delayed(get_polygon)(m, gx, gy, minx, miny) \
                                            for gx,gy in itertools.product(range(gridx), range(gridy)))
    parts = [i for i in parts if i is not None and aoi.intersects(i)]
    parts = gpd.GeoDataFrame(parts, columns=['geometry'], crs=epsg4326)
    print (f"\naoi covered by {len(parts)} chips")
    return parts


def select_partitions(orig_shapefile, aoi_wkt_file, aoi_name, tiles_name, dest_dir):
    """
    selects the geometries in 'orig_shafile' that have some intersention with aoi,
    assigns them an identifier and saves them in a new file.
    """
    print ("reading orig shapefile", flush=True)
    parts = gpd.read_file(orig_shapefile)
    if not  parts.crs == epsg4326:
           raise ValueError("'orig_shapefile' must be in epsg4326, lon/lat degrees "+\
                            f"but found \n{parts.crs}")
    
    with open(aoi_wkt_file, "r") as f:
        aoi = wkt.loads(f.read()) 
    
    print ("selecting geometries", flush=True)
    parts = [p for p in pbar(parts.geometry) if p.intersects(aoi)]
    if len(parts)==0:
        raise ValueError("no intersecting geometries found")
    # very small intersections probably are cause by numerical approximations
    # on the borders of the aoi
    parts = [p for p in parts if p.intersection(aoi).area>1e-5]
    
    parts = gpd.GeoDataFrame({'geometry': parts}, crs = CRS.from_epsg(4326))
    parts.to_file("/tmp/bb.geojson", driver='GeoJSON')
    parts = partitions.PartitionSet(aoi_name, data=parts)
    print ()
    parts.save_as(dest_dir, tiles_name)

    return parts

def zip_dataset(tiles_file, 
                foreign_tiles_file,
                images_dataset_def, 
                labels_dataset_def, 
                readme_file):
    

    images_dataset = utils.get_dataset_definition(images_dataset_def)
    images_dataset_name = images_dataset.get_dataset_name()

    if labels_dataset_def is not None:
        labels_dataset = utils.get_dataset_definition(labels_dataset_def)
        labels_dataset_name = labels_dataset.get_dataset_name()
    else:
        labels_dataset_def = None
        labels_dataset_name = None


    basedir = os.path.dirname(tiles_file)
    if basedir == "":
        basedir = "."
    filebase, _ = os.path.splitext(tiles_file)
    aoi_name = os.path.basename(tiles_file).split("_")[0]
    splits_file = f'{basedir}/{filebase}_splits.csv'
    expanded_file = f'{basedir}/{filebase}_expanded.geojson'
    destination_dir = f"{basedir}/{aoi_name}_{images_dataset_name}"
    if labels_dataset_name is not None:
        destination_dir += f"_{labels_dataset_name}"

    if foreign_tiles_file is not None:
        s = foreign_tiles_file
        foreign_tiles_name = s[s.find('_partitions_')+len('_partitions_'):].split("_")[0]
    else:
        foreign_tiles_name = None
        
    print ("preparing folders")
    os.makedirs(f"{destination_dir}/data", exist_ok=True)

    def remove_hash(filename):
        filebase, extension = os.path.splitext(filename)
        if filebase.endswith('_splits'):
            r = "_".join(filebase.split("_")[:-2])+"_splits"+extension
        elif filebase.endswith('_expanded'):
            r = "_".join(filebase.split("_")[:-2])+"_expanded"+extension
        else:
            r = "_".join(filebase.split("_")[:-1])+extension            
        return r
    
    print ("creating expanded file for easy visualization of label proportions")
    p = partitions.PartitionSet.from_file(tiles_file)
    p.expand_proportions()

    # copy files
    print ("copying metadata files")
    for filename in [tiles_file, foreign_tiles_file, splits_file, expanded_file]:
        if filename is not None and os.path.isfile(filename):
            dest_file = f"{destination_dir}/{remove_hash(os.path.basename(filename))}"
            if dest_file.endswith('splits.csv'):
                # splits file is just called 'splits.csv'
                dest_file = f"{destination_dir}/splits.csv"

            shutil.copyfile(f"{filename}", dest_file)

            # remove columns from other datasets in tiles file
            if filename == tiles_file:
                m = gpd.read_file(dest_file)
                remove_columns = [c for c in m.columns if '_proportions' in c and labels_dataset_name is not None and not labels_dataset_name in c]
                if len(remove_columns)>0:
                    print ("removing data from other label datasets")
                    m = m[[c for c in m.columns if not c in remove_columns]]
                    m.to_file(dest_file, driver='GeoJSON')

    if readme_file is not None:
        shutil.copyfile(readme_file, f"{destination_dir}/README.txt")

    print ("reading tiles file")
    # read chip definitions
    c = gpd.read_file(tiles_file)
    # gather imgs, labels and proportions 
    partitionset_dir = os.path.splitext(f"{basedir}/{tiles_file}")[0]
    n_skipped_chips = 0
    n_included_chips = 0

    included_chipids = []

    for _,i in pbar(c.iterrows(), max_value=len(c)):
        r = {}
        img_filename   = f"{partitionset_dir}/{images_dataset_name}/{i.identifier}.tif"
        if labels_dataset_name is not None:
            label_filename = f"{partitionset_dir}/{labels_dataset_name}/{i.identifier}.tif"
        if os.path.exists(img_filename):

            img = imread(img_filename).astype(np.int16)
            if 'map_values' in dir(images_dataset):
                img = images_dataset.map_values(img)
            
            coords = np.r_[i.geometry.envelope.boundary.coords]
            center_latlon = coords.mean(axis=0)[::-1]
            cmax = coords.max(axis=0)[::-1]
            cmin = coords.min(axis=0)[::-1]
            nw = np.r_[cmax[0], cmin[1]]
            se = np.r_[cmin[0], cmax[1]]    

            r['chip'] = img
            r['chip_id'] = i.identifier
            r['center_latlon'] = center_latlon
            r['corners'] = { 'nw': nw, 'se': se }
            
            if labels_dataset_name is not None and os.path.exists(label_filename):
                label = imread(label_filename).astype(np.int16)
                if 'map_values' in dir(labels_dataset):
                    label = labels_dataset.map_values(label)

                r['label'] = label
                props = {}
                if f'{labels_dataset_name}_proportions' in i.keys():
                    props['partitions_aschip'] = i[f'{labels_dataset_name}_proportions'].copy()

                if foreign_tiles_name is not None and f'foreignid_{foreign_tiles_name}' in i.keys():
                    props[f'partitions_{foreign_tiles_name}'] = i[f'{labels_dataset_name}_proportions_at_{foreign_tiles_name}'].copy()
                    props[f'foreignid_{foreign_tiles_name}'] = i[f'foreignid_{foreign_tiles_name}']

                if len(props)>0:
                    r['label_proportions'] = props

            if labels_dataset is None or\
               'include_chip_in_dataset' not in dir(labels_dataset) or \
               labels_dataset.include_chip_in_dataset(r):
                with open(f"{destination_dir}/data/{i.identifier}.pkl", "wb") as f:
                    pickle.dump(r, f)
                n_included_chips += 1
                included_chipids.append(r['chip_id'])
            else:
                n_skipped_chips += 1

    print (f"including {n_included_chips} chips, and skipped {n_skipped_chips}")

    # remove usused chips from metadata files
    if n_included_chips < len(c):
        chip_metafiles = [i for i in os.listdir(destination_dir) if 'aschip' in i]
        for filename in chip_metafiles:
            print ("removing unused chips from", filename, flush=True)
            filename = f"{destination_dir}/{filename}"
            if filename.endswith("splits.csv"):
                f = pd.read_csv(filename)
                f = f[f.identifier.isin(included_chipids)]
                f.to_csv(filename, index=False)
            elif filename.endswith(".geojson"):
                f = gpd.read_file(filename)
                f = f[f.identifier.isin(included_chipids)]
                f.to_file(filename, driver='GeoJSON')
            else:
                continue

    # create zip file
    print ("zipping all content")

    # Create object of ZipFile
    zipfile = f"{destination_dir}.zip"
    with ZipFile(zipfile, 'w', compression=ZIP_DEFLATED, compresslevel=9) as zip_object:
        # Traverse all files in directory
        for folder_name, sub_folders, file_names in os.walk(destination_dir):
            for filename in file_names:
                # Create filepath of files in directory
                file_path = os.path.join(folder_name, filename)
                # Add files to zip file
                zip_object.write(file_path, file_path)

    if os.path.exists(zipfile):
        print(f"dataset zip file created: {zipfile}")
    else:
        raise ValueError(f"could not create zip file {zipfile}")

    print ("done!")