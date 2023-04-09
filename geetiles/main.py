import argparse
from .cmds import *
from . import __version__

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='commands', dest='cmd')
    
    grid_parser = subparsers.add_parser('grid', help='make a grid from an AOI')
    grid_parser.add_argument('--aoi_wkt_file', required=True, type=str, help='the file containing the AOI in lon/lat degrees')
    grid_parser.add_argument('--chip_size_meters', required=True, type=int, help='the length of each tile side in meters')
    grid_parser.add_argument('--aoi_name', type=str, required=True, help='the name of the AOI, used to name the resulting file')
    grid_parser.add_argument('--dest_dir', type=str, required=True, help='the folder to store the resulting file')

    rnd_parser = subparsers.add_parser('random', help='make a random partitions in the AOI')
    rnd_parser.add_argument('--aoi_wkt_file', required=True, type=str, help='the file containing the AOI in lon/lat degrees')
    rnd_parser.add_argument('--max_rectangle_size_meters', required=True, type=int, help='the max size of any tile in meters')
    rnd_parser.add_argument('--aoi_name', required=True, type=str, help='the name of the AOI, used to name the resulting file')
    rnd_parser.add_argument('--dest_dir', required=True, type=str, help='the folder to store the resulting file')

    sel_parser = subparsers.add_parser('select', help='selects the geometries in orig_shafile that have some intersention with aoi')
    sel_parser.add_argument('--aoi_wkt_file', required=True, type=str, help='the file containing the AOI in lon/lat degrees')
    sel_parser.add_argument('--orig_shapefile', required=True, type=str, help='the shapefile with the geometries to select')
    sel_parser.add_argument('--aoi_name', required=True, type=str, help='the name of the AOI, used to name the resulting file')
    sel_parser.add_argument('--dest_dir', required=True, type=str, help='the folder to store the resulting file')
    sel_parser.add_argument('--tiles_name', required=True, type=str, help='a name for the selected geometries, used to name the resulting file')

    dwn_parser = subparsers.add_parser('download', help='downloads tiles from gee.')
    dwn_parser.add_argument('--tiles_file', required=True, type=str, help='output file produced by grid, random or select commands. It requires columns "geometry" and "identifier", and be in crs epsg4326. Downloaded tiles will be stored as geotiffs alongside in the same folder.')
    dwn_parser.add_argument('--dataset_def', required=True, type=str, help="A file with python code defining a class with the dataset definition. See files under defs/ for examples . Can also be the string 'sentinel2-rgb-median-2020' or 'esa-world-cover' for built-in definitions")
    dwn_parser.add_argument('--pixels_lonlat', default=None, type=str, help='a tuple, if set, the tile will have this exact size in pixels, regardless the physical size. For instance --pixels_lonlat [100,100]')
    dwn_parser.add_argument('--meters_per_pixel', default=None, type=int, help='an int, if set, the tile pixel size will be computed to match the requested meters per pixel. You must use exactly one of --meters_per_pixel or --pixels_lonlat.')
    dwn_parser.add_argument('--max_downloads', default=None, type=str, help='max number of tiles to download.')
    dwn_parser.add_argument('--shuffle', default=False, action='store_true', help='if set, the order of tile downloading will be shuffled.')
    dwn_parser.add_argument('--skip_if_exists', default=False, action='store_true', help='if set, tiles already existing in the destination folder will not be downloaded.')
    dwn_parser.add_argument('--skip_confirm', default=False, action='store_true', help='if set, proceeds with no user confirmation.')
    dwn_parser.add_argument('--ee_auth_mode', default=None, type=str, help='gee auth mode, see https://developers.google.com/earth-engine/apidocs/ee-authenticate.')
    dwn_parser.add_argument('--n_processes', default=10, type=int, help='number of parallel processes.')

    int_parser = subparsers.add_parser('intersect', help='for each tile, it gets the largest tile id from another set intersecting it.')
    int_parser.add_argument('--tiles_file', required=True, type=str, help='output file produced by grid, random or select commands. It requires columns "geometry" and "identifier", and be in crs epsg4326.')
    int_parser.add_argument('--foreign_tiles_file', required=True, type=str, help='the other tile set.')

    lpc_parser = subparsers.add_parser('lp.compute', help='computes labels proportions from downloaded dataset.')
    lpc_parser.add_argument('--tiles_file', required=True, type=str, help='output file produced by grid, random or select commands. It requires columns "geometry" and "identifier", and be in crs epsg4326.')
    lpc_parser.add_argument('--labels_dataset_def', required=True, type=str, help='the dataset definition for which to compute proportions. Must have been downloaded previously. See "download --dataset-def".')

    lpf_parser = subparsers.add_parser('lp.from_foreign', help='computes labels proportions from another geometry partition on the same area by interseting geometries.')
    lpf_parser.add_argument('--tiles_file', required=True, type=str, help='output file produced by grid, random or select commands. It requires columns "geometry" and "identifier", and be in crs epsg4326.')
    lpf_parser.add_argument('--foreign_tiles_file', required=True, type=str, help='the tiles file from which labels will be taken to compute proportions by intersecting geometries.')
    lpf_parser.add_argument('--labels_dataset_def', required=True, type=str, help='the dataset definition for which to compute proportions. Must have been downloaded previously. See "download --dataset-def".')

    split_parser = subparsers.add_parser('split', help='splits geometries in train, test and val with contiguous bands.')
    split_parser.add_argument('--tiles_file', required=True, type=str, help='output file produced by grid, random or select commands. It requires columns "geometry" and "identifier", and be in crs epsg4326.')
    split_parser.add_argument('--nbands', required=True, type=int, help='the number of bands to create.')
    split_parser.add_argument('--angle', required=True, type=float, help='angle to use for the bands, in [-1.5707,1.5707] = [-pi/2, pi/2].')
    split_parser.add_argument('--train_pct', required=True, type=float, help='percentage of train data, in [0,1].')
    split_parser.add_argument('--test_pct', required=True, type=float, help='percentage of test data, in [0,1].')
    split_parser.add_argument('--val_pct', required=True, type=float, help='percentage of validation data, in [0,1].')
    split_parser.add_argument('--foreign_tiles_name', required=False, default=None, type=str, help='add a split that keeps tiles within coarser foreign tiles together in the same split. Must have run lp.from_foreign before.')

    zip_parser = subparsers.add_parser('zip.dataset', help='assembles chips into pkls and zips all data.')
    zip_parser.add_argument('--tiles_file', required=True, type=str, help='output file produced by grid, random or select commands.')
    zip_parser.add_argument('--foreign_tiles_file', default=None, required=False, type=str, help='the tiles file from which foreign label proportions were computed.')
    zip_parser.add_argument('--images_dataset_def', required=True, type=str, help="name of the dataset or python file used to download tiles (see 'dataset_def' in 'download').")
    zip_parser.add_argument('--labels_dataset_def', default=None, required=False, type=str, help="name of the dataset or python file used to download tiles (see 'dataset_def' in 'download').")
    zip_parser.add_argument('--readme_file', default=None, required=False, type=str, help="name of the README.txt file to add to the zip file.")

    print ("-----------------------------------------------------------")
    print (f"Google Earth Engine dataset extractor utility {__version__}")
    print ("-----------------------------------------------------------")
    print ()
    args = parser.parse_args()
    if args.cmd == 'grid':
        print ("making grid", flush=True)

        make_grid(aoi_wkt_file     = args.aoi_wkt_file, 
                  chip_size_meters = args.chip_size_meters, 
                  aoi_name         = args.aoi_name, 
                  dest_dir         = args.dest_dir)

    elif args.cmd == 'random':
        print ("making random partitions", flush=True)
        make_random_partitions(aoi_wkt_file              = args.aoi_wkt_file, 
                               max_rectangle_size_meters = args.max_rectangle_size_meters, 
                               aoi_name                  = args.aoi_name, 
                               dest_dir                  = args.dest_dir)
        
    elif args.cmd == 'select':
        print ("selecting partitions", flush=True)
        select_partitions(orig_shapefile = args.orig_shapefile,
                          aoi_wkt_file   = args.aoi_wkt_file, 
                          aoi_name       = args.aoi_name, 
                          tiles_name     = args.tiles_name,
                          dest_dir       = args.dest_dir)
        
    elif args.cmd == 'download':

        print ("downloading tiles from GEE")
        try:
            download(   tiles_file        = args.tiles_file, 
                        dataset_def       = args.dataset_def, 
                        pixels_lonlat     = args.pixels_lonlat, 
                        meters_per_pixel  = args.meters_per_pixel, 
                        max_downloads     = args.max_downloads,
                        shuffle           = args.shuffle,
                        skip_if_exists    = args.skip_if_exists,
                        ee_auth_mode      = args.ee_auth_mode,
                        skip_confirm      = args.skip_confirm, 
                        n_processes       = args.n_processes
                    )
        except ValueError as e:
            print ("ERROR.", e)
            quit(-1)

    elif args.cmd == 'intersect':
        print ("computing intersections with foreign tiles")
        intersect_with_foreign(tiles_file         = args.tiles_file,
                               foreign_tiles_file = args.foreign_tiles_file)

    elif args.cmd == 'lp.compute':
        print ("computing proportions")
        label_proportions_compute(tiles_file = args.tiles_file, 
                                  labels_dataset_def = args.labels_dataset_def)      

    elif args.cmd == 'lp.from_foreign':
        print ("computing proportions from foreign tiles")
        label_proportions_from_foreign(tiles_file         = args.tiles_file,
                                       foreign_tiles_file = args.foreign_tiles_file,
                                       labels_dataset_def = args.labels_dataset_def)  
        
    elif args.cmd == 'split':
        print ("splitting bands")
        split(tiles_file         = args.tiles_file, 
              nbands             = args.nbands, 
              angle              = args.angle,
              train_pct          = args.train_pct, 
              test_pct           = args.test_pct, 
              val_pct            = args.val_pct,
              foreign_tiles_name = args.foreign_tiles_name)
        
    elif args.cmd == 'zip.dataset':
        print ("zipping dataset")
        zip_dataset(tiles_file          = args.tiles_file, 
                    foreign_tiles_file  = args.foreign_tiles_file,
                    images_dataset_def  = args.images_dataset_def, 
                    labels_dataset_def  = args.labels_dataset_def, 
                    readme_file         = args.readme_file)