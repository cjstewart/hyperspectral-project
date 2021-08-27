# libraries
# data
import pandas as pd
import numpy as np
import h5py
from PIL import Image
import json
import pystac
from pystac.extensions.eo import EOExtension
from pystac.extensions.eo import Band
from pystac.extensions.view import ViewExtension
from pystac.extensions.sat import SatExtension
from pystac.extensions.eo import EOExtension
from pystac.extensions.projection import ProjectionExtension

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ml
import sklearn as skl
import skimage.measure
import skimage.transform

# utilites
import sys
import os
from pathlib import Path
import datetime
import random
import string
import math
import tqdm

# dataframe options
pd.set_option('display.max_columns', None)

#%matplotlib inline

#import warnings
#warnings.filterwarnings('ignore')

# image manipulation
from osgeo import gdal
gdal.UseExceptions()
from shapely.geometry import Polygon, mapping
from pyproj import CRS
from pyproj import Transformer


# error logging
import logging
downSampler_logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------------------
def find_files(data_dir_path):
    """
    This function takes in a path to a data directory, walks the directory and
    returns a dictionary of the filenames and paths of all the data files.
    """
    downSampler_logger.info("find_files function called")  # logging
    # get all the h5 data filenames and paths
    file_dict = {} # to store file:path pairs
    print('Finding data files...')
    for p, _, file in os.walk(data_dir_path): # get all the data files and their paths
        for f in file:
            if f.endswith(".h5"): # only .h5 files read in
                file_dict[f] = os.path.join(p, f) # if you want a string path
                #file_dict[f] = Path(os.path.join(p, f)) # if you want a Path object path

    return file_dict

# ----------------------------------------------------------------------------------------------------------------------------------------

# HDF5 file structure map from this SO answer: https://stackoverflow.com/a/53340677
# Much more better!
def descend_obj(obj,sep='\t'):
    """
    Helper function that recursively iterates through groups in a HDF5 file and
    prints the groups and datasets names and datasets attributes
    --------
    Parameters
        obj -- file object to parse the structure of
    --------
    Returns
    --------
    None
    """
    downSampler_logger.info("descend_obj function called") # logging

    if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
        for key in obj.keys():
            print(sep,'-',key,':',obj[key])
            descend_obj(obj[key],sep=sep+'\t')
    elif type(obj)==h5py._hl.dataset.Dataset:
        for key in obj.attrs.keys():
            print(sep+'\t','-',key,':',obj.attrs[key])

def h5dump(path,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    downSampler_logger.debug("h5dump function called") # logging

    with h5py.File(path,'r') as f:
         descend_obj(f[group])

# ----------------------------------------------------------------------------------------------------------------------------------------

def h5data2array(data_file_path, clamp_values=True):
    """
    h5dataset2array reads in a hdf5 file and extracts and returns:
        1. reflectance array (with the no data value and reflectance scale factor applied)
        2. dictionary of metadata including spatial information, and wavelengths of the bands
    --------
    Parameters
        data_file_path -- full or relative path and name of reflectance hdf5 file
    --------
    Returns
    --------
    reflArray:
        array of reflectance values
    metadata:
        dictionary containing the following metadata:
            bands: # of bands (float)
            data ignore value: value corresponding to no data (float)
            epsg: coordinate system code (float)
            map info: coordinate system, datum & ellipsoid, pixel dimensions, and origin coordinates (string)
            reflectance scale factor: factor by which reflectance is scaled (float)
            wavelength: wavelength values (float)
            wavelength unit: 'nm' (string)
    --------
    Example Execution:
    --------
    refl_clean, wavelength_array, FWHM_array, metadata = h5data2array("myHDF5file.h5")

    """
    downSampler_logger.info("h5data2array function called") # logging


    hdf5_file = h5py.File(data_file_path,'r') # open file
    #Get the site name
    file_attrs_str = str(list(hdf5_file.items())).split("'")
    sitename = file_attrs_str[1]

    # Extract the reflectance & wavelength datasets
    refl_obj = hdf5_file[sitename]['Reflectance']
    refl_data = refl_obj['Reflectance_Data'] # reflectance dataset

    # Create dictionary containing relevant metadata information
    metadata_dict = {}
    metadata_dict['map info'] = refl_obj['Metadata']['Coordinate_System']['Map_Info']
    metadata_dict['wavelength'] = refl_obj['Metadata']['Spectral_Data']['Wavelength']

    # Extract no data value & scale factor
    metadata_dict['data ignore value'] = float(refl_data.attrs['Data_Ignore_Value'])
    metadata_dict['reflectance scale factor'] = float(refl_data.attrs['Scale_Factor'])
    metadata_dict['Spatial_Resolution_X_Y'] = refl_data.attrs['Spatial_Resolution_X_Y']

    # Apply no data value
    refl_array = np.array(refl_data)
    arr_size = refl_data.shape
    if np.isin(refl_array, metadata_dict['data ignore value']).any():
        print('% No Data: ',np.round(np.count_nonzero(refl_array == metadata_dict['data ignore value'])*100/(arr_size[0]*arr_size[1]*arr_size[2]),1)) # calculates % of missing data values
        refl_array[np.where(refl_array == metadata_dict['data ignore value'])] = 0.0 # use float instead #np.nan # replace no data values with NaNs

    # Apply scale factor
    refl_array = refl_array/metadata_dict['reflectance scale factor']

    # metadata
    # extract wavelength bands and FWHMs
    wavelength_array = refl_obj['Metadata']['Spectral_Data']['Wavelength'][...] # wavelength dataset
    FWHM_array = refl_obj['Metadata']['Spectral_Data']['FWHM'][...] # FWHM dataset

    # Extract spatial extent from attributes
    metadata_dict['spatial extent'] = refl_data.attrs['Spatial_Extent_meters']

    # Extract projection information
    metadata_dict['Proj4'] = refl_obj['Metadata']['Coordinate_System']['Proj4'][...]
    metadata_dict['EPSG Code'] = refl_obj['Metadata']['Coordinate_System']['EPSG Code'][...]
    metadata_dict['Map_Info'] = refl_obj['Metadata']['Coordinate_System']['Map_Info'][...]
    metadata_dict['Coordinate_System_String'] = refl_obj['Metadata']['Coordinate_System']['Coordinate_System_String'][()] # Coordinate System string

    # Extract map information: spatial extent & resolution (pixel size)
    # mapInfo = refl_obj['Metadata']['Coordinate_System']['Map_Info'][...] # I don't think this is used... remove later if not

    hdf5_file.close # close file

    # clamping function for values outside the acceptable range
    def clamp(array, low, high):
        array[array < low] = low
        array[array > high] = high
        return array

    if clamp_values:
        refl_clean = clamp(refl_array, 0, 1) # first clamp to [0,1] range
        return refl_clean, wavelength_array, FWHM_array, metadata_dict
    else:
        return refl_array, wavelength_array, FWHM_array, metadata_dict

# ----------------------------------------------------------------------------------------------------------------------------------------

def array2h5data(refl_array, wavelength_array, FWHM_array, metadata_dict, filename_output):
    """
    Takes in a 3-D reflectance array, an array of band centre wavelengths, FWHM array,
    and an additional metadata dictionary and generates a HDF5 file with the given filename.
    """
    downSampler_logger.info("array2h5data function called") # logging


    # scale the reflectance data up by the original reflectance factor to save disk space
    scale_fac = metadata_dict['reflectance scale factor']
    refl_array = refl_array*scale_fac

    hf = h5py.File(filename_output, 'w') # create hdf5 file
    g1 = hf.create_group('Reflectance') # create main group
    g2 = hf.create_group('Reflectance/Metadata') # group for metadata

    # datasets
    # ------------------------------------------------------------------------------
    # reflectance data
    refl_dset = g1.create_dataset('Reflectance_Data',data=refl_array, dtype='i') # dataset for reflectance data
    refl_dset.attrs['Description'] = 'Atmospherically corrected reflectance.'
    refl_dset.attrs['data ignore value'] = metadata_dict['data ignore value']
    refl_dset.attrs['reflectance scale factor'] = metadata_dict['reflectance scale factor']
    refl_dset.attrs['Spatial_Resolution_X_Y'] = metadata_dict['Spatial_Resolution_X_Y']
    refl_dset.attrs['spatial extent'] = metadata_dict['spatial extent']

    # wavelength data
    wav_dset = g2.create_dataset('Wavelength',data=wavelength_array) # band centre wavelength data
    wav_dset.attrs['Description'] = 'Central wavelength of the reflectance bands.'
    wav_dset.attrs['Units'] = 'nanometers'

    # FWHM data
    FWHM_dset = g2.create_dataset('FWHM',data=FWHM_array) # FWHM data
    FWHM_dset.attrs['Description'] = 'Full width half maximum of reflectance bands.'
    FWHM_dset.attrs['Units'] = 'nanometers'


    hf.close() # close file to save and write to disk


# ----------------------------------------------------------------------------------------------------------------------------------------


def array2h5raster(refl_array, wavelength_array, FWHM_array, metadata_dict, filename_output):
    """
    Takes in a 3-D reflectance array, an array of band centre wavelengths, FWHM array,
    and an additional metadata dictionary and generates a HDF5 file with the given filename.
    Each band of reflectance data is written to its own hdf5 dataset.
    """
    downSampler_logger.info("array2h5raster function called") # logging

    # scale the reflectance data up by the original reflectance factor to save disk space
    scale_fac = metadata_dict['reflectance scale factor']
    refl_array = refl_array*scale_fac
    wavelength_array = np.array(wavelength_array)*1000 # convert to nm
    FWHM_array = np.array(FWHM_array)*1000

    hf = h5py.File(filename_output, 'w') # create hdf5 file
    g1 = hf.create_group('Reflectance') # create main group
    g2 = hf.create_group('Reflectance/Metadata') # group for metadata

    # datasets
    # ------------------------------------------------------------------------------
    # reflectance data

    for band in range(refl_array.shape[2]):
        #print("band",band)
        band_name = 'Band_' + str(band+1).zfill(3) # make into e.g. 001 or 013 instead of 1 or 13 etc.
        refl_dset = g1.create_dataset(band_name,data=refl_array[:,:,band], dtype='i') # dataset for each band of reflectance data
        refl_dset.attrs['Description'] =  'Atmospherically corrected reflectance.'
        refl_dset.attrs['Wavelength'] = wavelength_array[band]# wavelength in nm
        refl_dset.attrs['Wavelength_units'] = 'nm' # wavelength units
        refl_dset.attrs['data ignore value'] = metadata_dict['data ignore value']
        refl_dset.attrs['reflectance scale factor'] = metadata_dict['reflectance scale factor']
        refl_dset.attrs['Spatial_Resolution_X_Y'] = metadata_dict['Spatial_Resolution_X_Y']
        refl_dset.attrs['spatial extent'] = metadata_dict['spatial extent']

    # wavelength data
    wav_dset = g2.create_dataset('Wavelength',data=wavelength_array) # band centre wavelength data
    wav_dset.attrs['Description'] = 'Central wavelength of the reflectance bands.'
    wav_dset.attrs['Units'] = 'nanometers'

    # FWHM data
    FWHM_dset = g2.create_dataset('FWHM',data=FWHM_array) # FWHM data
    FWHM_dset.attrs['Description'] = 'Full width half maximum of reflectance bands.'
    FWHM_dset.attrs['Units'] = 'nanometers'

    # coordinates data
    g2_1 = g2.create_group('Coordinate_System') # group for metadata
    Proj4_dset = g2_1.create_dataset('Proj4',data=metadata_dict['Proj4']) # band Proj4 data
    EPSG_dset = g2_1.create_dataset('EPSG Code',data=metadata_dict['EPSG Code']) # EPSG Code
    map_dset = g2_1.create_dataset('Map_Info',data=metadata_dict['Map_Info']) # Map Info
    coor_dset = g2_1.create_dataset('Coordinate_System_String',data=metadata_dict['Coordinate_System_String']) # coordinate system
    map_dset.attrs['Description'] = ("List of geographic information in the following order:\n"
                                    "   - Projection name\n"
                                    "   - Reference (tie point) pixel x location (in file coordinates)\n"
                                    "   - Reference (tie point) pixel y location (in file coordinates)\n"
                                    "   - Pixel easting\n"
                                    "   - Pixel northing\n"
                                    "   - x pixel size\n"
                                    "   - y pixel size\n"
                                    "   - Projection zone (UTM only)\n"
                                    "   - North or South (UTM only)\n"
                                    "   - Datum\n"
                                    "   - Units\n"
                                    "   - Rotation Angle\n"
                                    )

    #     - Description : List of geographic information in the following order:
    #         - Projection name
    #         - Reference (tie point) pixel x location (in file coordinates)
    #         - Reference (tie point) pixel y location (in file coordinates)
    #         - Pixel easting
    #         - Pixel northing
    #         - x pixel size
    #         - y pixel size
    #         - Projection zone (UTM only)
    #         - North or South (UTM only)
    #         - Datum
    #         - Units
    #         - Rotation Angle

    hf.close() # close file to save and write to disk


# ----------------------------------------------------------------------------------------------------------------------------------------


def array2gtiff_raster(refl_array, wavelength_array, FWHM_array, metadata_dict, pixelWidth, pixelHeight, file_path, filename_prefix="WYVERN_DS"):
    """
    Takes in a 3-D reflectance array, an array of band centre wavelengths, FWHM array,
    and an additional metadata dictionary and generates a HDF5 file with the given filename.
    Each band of reflectance data is written to its own geotiff image.

    Parameters
    -----------
    refl_array : 3-D array_like
                Array of image spectral reflectance values that will be used in the raster layers

    wavelength_array : 1-D array_like
                Array of the band centre wavelengths for the given spectral reflectance array

    FWHM_array : 1-D array_like
                Array of the band widths

    metadata_dict : dictionary of metadata

    pixelWidth : int or float
                Ground coverage width of a pixel in meters

    pixelHeight : int or float
                Ground coverage height of a pixel in meters

    file_path : string or path object
                Directory path to save outputted files to

    filename_prefix : string, default = "WYVERN_DS"
                Prefix to append to front of outputted filenames
    --------
    Example Execution:
    --------
    array2gtiff_raster(refl_array, wavelength_array, FWHM_array, metadata_dict, filename_output, pixelWidth, pixelHeight)

    """
    downSampler_logger.info("array2gtiff_raster function called") # logging

    # set up parameters
    rows = refl_array.shape[0] # rows of image
    cols = refl_array.shape[1] # columns of image
    num_bands = refl_array.shape[2] # number of bands
    NO_DATA = metadata_dict['data ignore value'] # -9999
    dtype = gdal.GDT_Float32

    coords_utm = [] # input UTM coordinates
    coords_utm.append((metadata_dict['spatial extent'][0],metadata_dict['spatial extent'][2]))
    coords_utm.append((metadata_dict['spatial extent'][0],metadata_dict['spatial extent'][3]))
    coords_utm.append((metadata_dict['spatial extent'][1],metadata_dict['spatial extent'][3]))
    coords_utm.append((metadata_dict['spatial extent'][1],metadata_dict['spatial extent'][2]))

    driver = gdal.GetDriverByName('GTiff')

    # write out raster bands
    for band in range(num_bands):
        band_name = filename_prefix + '_band_' + str(band+1).zfill(3) + '.tif' # make into e.g. 001 or 013 instead of 1 or 13 etc.
        full_path = str(os.path.join(file_path, band_name)) # gdal can't seem to handle filepaths

        out_raster = driver.Create(full_path, cols, rows, 1, dtype)
        projection = 'EPSG:' + metadata_dict['EPSG Code'].item().decode('UTF-8') # UTM coordinate system
        out_raster.SetProjection(projection) # apply CRS
        geo_transform = (coords_utm[1][0], pixelWidth, 0, coords_utm[1][1], 0, -pixelHeight) # geo_transform = (x top left, x cell size, x rotation, y top left, y rotation, negative y cell size)
        out_raster.SetGeoTransform(geo_transform)

        outband = out_raster.GetRasterBand(1)
        outband.WriteArray(refl_array[:,:,band]) # write reflectance data to band
        outband.SetNoDataValue(NO_DATA)
        outband.FlushCache() # save to disk
        outband = None # free up memory
        out_raster = None


# ----------------------------------------------------------------------------------------------------------------------------------------


def reband_spectral_array(spect_array, input_bandcentres_array, output_bandcentres_array):
    """
    Takes in both a 1D spectral array and a input array of band centres and then returns the linearly interpolated
    reflectance values of the desired output bands.
    WARNING: input and output band centre wavelengths are all in nm units!!

    Parameters
    -----------
    spect_array : 1-D array_like
                Array of image spectral reflectance values that will be used as the basis for rebanding

    input_bandcentres_array : 1-D array_like
                Array of the band centre wavelengths for the given spectral reflectance array

    output_bandcentres_array : 1-D array_like
                Array of the desired new band centre wavelengths
    --------
    Example Execution:
    --------
    refl_array_new = reband_spec(spect_array, input_bandcentres_array, output_bandcentres_array)

    """
    #downSampler_logger.info("reband_spectral_array function called") # logging


    if len(input_bandcentres_array) == len(spect_array): # error checking
        downSampler_logger.debug("reband_spectral_array returns: {}".format(np.interp(x=output_bandcentres_array, xp=input_bandcentres_array, fp=spect_array, left=None, right=None, period=None))) # logging
        return np.interp(x=output_bandcentres_array, xp=input_bandcentres_array, fp=spect_array, left=None, right=None, period=None)
    else:
        print("Error: input spectral array must be same length as input band centres array")
        return []


# ----------------------------------------------------------------------------------------------------------------------------------------


# function to create band widths array
def band_widths(bandcentres_array):
    """
    Generates band widths from given band centre wavelengths
    """
    downSampler_logger.info("band_widths function called") # logging

    bandwidths_array = np.empty(len(bandcentres_array)) # create empty array to store bands widths

    for i in range(0,len(bandcentres_array)-1): # calculate band width as difference between band centres (approximation)
        bandwidths_array[i] = (bandcentres_array[i+1] - bandcentres_array[i])

    # handle last index case
    bandwidths_array[-1] = (bandcentres_array[-1] - bandcentres_array[-2])

    return bandwidths_array


# ----------------------------------------------------------------------------------------------------------------------------------------


def downSample_reband_array(img_array, GSD_input, GSD_output, input_bandcentres_array, output_bandcentres_array):
    """
    Returns a spatially downsampled (x,y dim) 3-D hyperspectral image array
    which has also has its spectral dimension (z) rebanded to a desired set of band centres.

    Parameters
    -----------
    img_array : 3-D array_like
                Array of image reflectance data - [x,y] dimensions are spatial and [z] dimension is the spectral component.

    GSD_input : integer or float
                The Ground Sample Distance (GSD) of the input image in meters.
                If a float is provided the actual scaling factor will be an integer of the rounded ratio of GSD_output/GSD_input.

    GSD_output : integer or float
                The desired Ground Sample Distance (GSD) of the ouput image in meters.
                If a float is provided the actual scaling factor will be an integer of the rounded ratio of GSD_output/GSD_input.

    input_bandcentres_array : array_like
                The spectral band centres of the input image array.

    output_bandcentres_array : array_like
                The desired spectral band centres of the output image array.


    Returns
    -------
    image : ndarray
        Down-sampled image with the same [x,y] dimensions and a rebanded [z] dimension


    Example Execution:
    --------
    img_array_ds = downSample_array(img_array, 1, 2) # downsample image by factor of 2

    """
    downSampler_logger.info("downSample_reband_array function called") # logging
    downSampler_logger.debug("GSD_output passed: {}, GSD_input passed: {}".format(GSD_output, GSD_input)) # logging

    downsample_factor = max(int(math.ceil(GSD_output/GSD_input)),1) # calculate the downsampling factor - outputs only integers!
    rescale_factor = float(downsample_factor)
    downSampler_logger.debug("Rescale_factor calculated: {}".format(rescale_factor)) # logging

    img_array_ds = skimage.measure.block_reduce(img_array[:, :, :],
                                             block_size=(downsample_factor, downsample_factor, 1),
                                             func=np.mean) # downsample image array
    downSampler_logger.debug("img_array shape passed: {}, img_array_ds shape downsampled: {}".format(img_array.shape, img_array_ds.shape)) # logging
    downSampler_logger.debug("img_array: {}, img_array_ds downsampled: {}".format(img_array, img_array_ds)) # logging

    # create empty array to store the rebanded spectral arrays
    #output_bandcentres_array = np.array(output_bandcentres_array)*1000 # convert to nm
    rebanded_array = np.zeros((img_array_ds.shape[0], img_array_ds.shape[1], len(output_bandcentres_array)))
    downSampler_logger.debug("Should be empty - rebanded_array shape: {}".format(rebanded_array.shape)) # logging
    downSampler_logger.debug("Should be empty - rebanded_array: {}".format(rebanded_array)) # logging


    # loop through all spatial pixels and reband the spectral component of each one
    downSampler_logger.info("rebrand_spectral _array function going to run {} times".format(img_array_ds.shape[0]*img_array_ds.shape[1])) # logging
    for x in range(0,img_array_ds.shape[0]): # x dimension
        for y in range(0,img_array_ds.shape[1]): # y dimension

            downSampler_logger.debug("Passed to reband_spectral_array - img_array_ds[x, y, :]: {}, \ninput_bandcentres_array: {}, \noutput_bandcentres_array: {}".format(img_array_ds[x, y, :], input_bandcentres_array, output_bandcentres_array)) # logging
            rebanded_array[x, y, :] = reband_spectral_array(img_array_ds[x, y, :],
                                                            input_bandcentres_array,
                                                            output_bandcentres_array) # build up empty array with rebanded arrays
    downSampler_logger.info("rebrand_spectral _array function run complete") # logging

    downSampler_logger.debug("Should be full now - rebanded_array shape: {}, rebanded_array: {}".format(rebanded_array.shape, rebanded_array)) # logging
    # upscale image back to original input image size (interpolating pixels in between)
    downSampler_logger.debug("Rescale of the rebanded_array shape: {}, rescaled rebanded_array: {}".format(skimage.transform.rescale(rebanded_array,(rescale_factor,rescale_factor,1.0)).shape, skimage.transform.rescale(rebanded_array,(rescale_factor,rescale_factor,1.0)))) # logging
    #return np.round(skimage.transform.rescale(rebanded_array,(rescale_factor,rescale_factor,1.0)),4) # scale back up to original size (interpolates)
    return np.round(rebanded_array,4) # round to 4 decimals

# ----------------------------------------------------------------------------------------------------------------------------------------


def toRGB(refl_array, filename_output, mode=1):
    """
    This is a rough way to convert visualize parts of a hyperspectrum and show them in RGB colour images
    This tool was helpful: https://academo.org/demos/wavelength-to-colour-relationship/
    and this resource: https://towardsdatascience.com/image-processing-with-python-5b35320a4f3c
    -----------------------------------------------------------------------------------------------------
    red ~= 625–740 nm -> pick 700 nm -> rgb(255,0,0) : wavelength_array[64] = 702.223
    green ~= 495–570 nm -> pick 510 nm -> rgb(0,255,0) : wavelength_array[26] = 511.8519
    blue ~= 450-495 nm -> pick 440 nm - > rgb(0,0,255) : wavelength_array[12] = 441.7151

    mode 2: new bands
    # 0.505, 0.526, 0.544, 0.565, 0.586, 0.606, 0.626, 0.646,  0.665, 0.682, 0.699, 0.715, 0.730, 0.745, 0.762, 0.779, 0.787, 0.804
    red ~= 625–740 nm -> pick 700 nm -> rgb(255,0,0) : wavelength_array[10] = 699
    green ~= 495–570 nm -> pick 510 nm -> rgb(0,255,0) : wavelength_array[0] = 505, pick 544 wavelength_array[4] instead
    blue ~= 450-495 nm -> pick 440 nm - > rgb(0,0,255) : wavelength_array[0] = 505 pick  wavelength_array[1]
    """
    downSampler_logger.info("toRGB function called") # logging
    if (mode == 1):
        # this mode is for original 426 spectrum array
        red = refl_array[:,:,64] + refl_array[:,:,64-1] + refl_array[:,:,64+1]
        green = refl_array[:,:,26] + refl_array[:,:,26-1] + refl_array[:,:,26+1]
        blue = refl_array[:,:,12] + refl_array[:,:,12-1] + refl_array[:,:,12+1]

    else:
        # this mode is for downsampled 18 spectrum array
        red = refl_array[:,:,10] + refl_array[:,:,10-1] + refl_array[:,:,10+1]
        green = refl_array[:,:,4] + refl_array[:,:,4-1] + refl_array[:,:,4+1]
        blue = refl_array[:,:,1] + refl_array[:,:,1-1] + refl_array[:,:,1+1]



    # clamping function for values outside the acceptable range
    def clamp(array, low, high):
        array[array < low] = low
        array[array > high] = high
        return array

    red_clampd = clamp(red, 0.0, 1.0)
    green_clampd = clamp(green, 0.0, 1.0)
    blue_clampd = clamp(blue, 0.0, 1.0)

    colour_array = np.stack((red_clampd, green_clampd, blue_clampd), axis=-1)

    im = Image.fromarray(np.uint8(colour_array*255))

    plt.imshow(np.asarray(im))
    plt.show()

    im.save(filename_output) # save file to .png
    #im.show()

# ----------------------------------------------------------------------------------------------------------------------------------------

def rand_string(length):
    '''
    Generates a string of random characters of the passed in length
    '''
    downSampler_logger.info("rand_string function called") # logging
    # from here: https://www.askpython.com/python/examples/generate-random-strings-in-python
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(length))

# ----------------------------------------------------------------------------------------------------------------------------------------


def metadata2geojsonSTAC(refl_array, wavelength_array, FWHM_array, metadata_dict, common_band_names, timestamp, start_timestamp, end_timestamp, file_path, filename_prefix="WYVERN_DS"):
    #resamp_refl_array, desired_band_centres, rebanded_FWHM_array, resamp_metadata_dict, common_band_names, timestamp, start_timestamp, end_timestamp, output_dir, filename_prefix

    """
    Takes in a 3-D reflectance array, an array of band centre wavelengths, FWHM array,
    and an additional metadata dictionary and generates a STAC metadata with the given filename.
    Each band of reflectance data is written to its own geotiff image.

    Parameters
    -----------
    refl_array : 3-D array_like
                Array of image spectral reflectance values that will be used in the raster layers

    wavelength_array : 1-D array_like
                Array of the band centre wavelengths for the given spectral reflectance array

    FWHM_array : 1-D array_like
                Array of the band widths

    metadata_dict : dictionary with metadata values

    --------
    Example Execution:
    --------
    writeMetadata_STAC(refl_array, wavelength_array, FWHM_array, metadata_dict, filename_output, pixelWidth, pixelHeight)

    """
    downSampler_logger.info("metadata2geojsonSTAC function called") # logging

    # create the catalog
    catalog = pystac.Catalog(id='catalog', description='Simulated satellite data catalog.')

    # params for STAC item
    # ------------------------------------------------------------------------------------

    # 1. lon/lat coordinates
    epsg_code = metadata_dict['EPSG Code'].item().decode('UTF-8')
    utm_crs = CRS.from_epsg(epsg_code) # UTm coordinate system
    latlon_crs = CRS.from_epsg('4326') # lat/lon coordinate system

    coords_utm = [] # input UTM coordinates
    coords_utm.append((metadata_dict['spatial extent'][0],metadata_dict['spatial extent'][2]))
    coords_utm.append((metadata_dict['spatial extent'][0],metadata_dict['spatial extent'][3]))
    coords_utm.append((metadata_dict['spatial extent'][1],metadata_dict['spatial extent'][2]))
    coords_utm.append((metadata_dict['spatial extent'][1],metadata_dict['spatial extent'][3]))

    transformer = Transformer.from_crs(utm_crs, latlon_crs) #transformer from UTM to lat/lon
    lats = [] # output lat/lon coordinates
    lons = []

    for coords in coords_utm:
        points = transformer.transform(coords[0],coords[1])
        lats.append(points[0])
        lons.append(points[1])

    coords_latlon = list(zip(lats,lons))

    bounds_right = max(lons)
    bounds_left = min(lons)
    bounds_top = max(lats)
    bounds_bottom = min(lats)

    bbox = [bounds_left, bounds_bottom, bounds_right, bounds_top] # image extent bounding box
    polygon = mapping(Polygon([
                        [bounds_left, bounds_bottom],
                        [bounds_left, bounds_top],
                        [bounds_right, bounds_top],
                        [bounds_right, bounds_bottom]
                    ]))
    # --------------------------------------------------------------------------------------

    # 2. Add the main STAC item
    item = pystac.Item(id=filename_prefix,
                     geometry=polygon,
                     bbox=bbox,
                     datetime=timestamp,
                     properties={
                                "license": "proprietary",
                                #"datetime": "2021-12-09T21:33:31.158193", # are these the
                                #"start_datetime": "2021-12-09T21:33:29.885445",
                                #"end_datetime": "2021-12-09T21:33:32.430941",
                                "datetime": str(timestamp),
                                "start_datetime": str(start_timestamp),
                                "end_datetime": str(end_timestamp),
                                "providers": [{
                                        "name": "Wyvern Space",
                                        "roles": [
                                            "producer",
                                            "licensor"
                                        ],
                                        "url": "https://www.wyvern.space/"
                                    }
                                ],
                                "platform": "Dragonette-Sim",
                                "constellation": "imagineDragonette",
                                "instruments": ["Hyperspectral-imager"],
                                "sensor_mode": "strip",
                                "sensor_type": "optical",
                                "product_type": "hyperspectral",
                                "created": "2021-05-04T00:00:01Z", # are these meant to be when the product was first created? or the satellite launched/operational?
                                "updated": "2021-05-05T00:30:55Z",
                                "gsd": metadata_dict['Spatial_Resolution_X_Y'][0],
                                "sat:orbit_state": "descending",
                                "sat:relative_orbit": 1,
                     },
                     stac_extensions = [
                        "https://stac-extensions.github.io/eo/v1.0.0/schema.json",
                        "https://stac-extensions.github.io/view/v1.0.0/schema.json",
                        "https://stac-extensions.github.io/sat/v1.0.0/schema.json",
                        "https://stac-extensions.github.io/projection/v1.0.0/schema.json"
                    ]
                      )

    catalog.add_item(item)

    # ------------------------------------------------------------------------------
    # 3. Set extension parameters
    item_ext = ViewExtension.ext(item)
    # 3.1 - view parameters - no data yet
    item_ext.sun_azimuth = -9999
    item_ext.sun_elevation = -9999
    item_ext.off_nadir = -9999
    item_ext.incidence_angle = -9999
    item_ext.azimuth = -9999
    #item.ext.view.azimuth = -9999

    # 3.2 - sat parameters
    #item.ext.sat.orbit_state = "descending"
    item_ext = SatExtension.ext(item)
    #item_ext.orbit_state = "descending"
    #item_ext.relative_orbit = 9999
    #item.ext.sat.relative_orbit = 9999

    # 3.3 - eo parameters
    item_ext = EOExtension.ext(item)
    item_ext.cloud_cover = -9999


    # 3.4 - proj parameters
    item_ext = ProjectionExtension.ext(item)
    try:
        item_ext.epsg = int(epsg_code)
    except ValueError:
        item_ext.epsg = None # non-valid epsg code!
    item_ext.shape = [refl_array.shape[1], refl_array.shape[0]] # raster shape in Y, X

    # -----------------------------------------------------------------------------------
    # 4. Create bands info on EO

    # desired_band_centres = [0.505, 0.526, 0.544, 0.565, 0.586,
    #                         0.606, 0.626, 0.646, 0.665, 0.682, 0.699,
    #                         0.715, 0.730, 0.745, 0.762, 0.779, 0.787, 0.804
    #                        ]
    #     common_band_names = ['green1','green2','green3','green4','yellow',
    #                          'red1','red2','red3','red4','red5','red6',
    #                          'rededge1','rededge2','rededge3',
    #                          'nir1','nir2','nir3','nir4'
    #                         ]

    #fwhm = np.round(band_widths(desired_band_centres),3)
    img_bands = []

    for band in range(refl_array.shape[2]):

        img_bands.append(Band.create(
                    name='Band_'+str(band+1).zfill(3),
                    common_name = common_band_names[band],
                    center_wavelength = wavelength_array[band],
                    full_width_half_max = FWHM_array[band], #fwhm[band],
                    #gain = -9999,
                    #offset = -9999,
                    #esun = -9999,
                    #raster_width = refl_array.shape[0],
                    #raster_height = refl_array.shape[1]
                   ))

    item_ext.bands=img_bands

    # -----------------------------------------------------------------------------------
    # 5. Make the assets

    for band in range(refl_array.shape[2]):
        band_name =  'Band_' + str(band+1).zfill(3) # make into e.g. 001 or 013 instead of 1 or 13 etc.
        file_name = filename_prefix + '_' + 'band_' + str(band+1).zfill(3) + '.tif'
        #print('creating asset for ' + band_name) # change to logging
        #downSampler_logger.debug('creating asset for ' + band_name) # logging

        # add the STAC assets
        item.add_asset(
                        key = band_name,
                        asset = pystac.Asset(
                            title = common_band_names[band],
                            #href = os.path.join(file_path, filename_prefix),
                            href = file_name, # no full filepath needed here since the metadata file is in the same directory as the image assets
                            media_type=pystac.MediaType.GEOTIFF,
                            roles = ["reflectance", "data"]
                        )
        )


    # 6. Add the preview asset
    item.add_asset(
                    key = filename_prefix+'_preview',
                    asset = pystac.Asset(
                        title = 'Preview',
                        #href = os.path.join(img_path, file_name),
                        href = filename_prefix+'_preview',
                        media_type=pystac.MediaType.PNG,
                        roles = ["overview"]
                    )
    )

    # 7. Add the thumbnail asset
    item.add_asset(
                    key = filename_prefix+'_thumbnail',
                    asset = pystac.Asset(
                        title = 'Thumbnail',
                        #href = os.path.join(img_path, file_name),
                        href = filename_prefix+'_thumbnail',
                        media_type=pystac.MediaType.PNG,
                        roles = ["overview"]
                    )
    )



    #catalog.normalize_hrefs(os.path.join(img_path, 'stac'))
    catalog.normalize_hrefs(file_path)
    catalog.make_all_asset_hrefs_relative()
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    #print(json.dumps(item.to_dict(), indent=4))

    # ----------------------------------------------------------------------------------------------------------------------------------------------
    # preview image
    print("Generating preview image...")
    toRGB(refl_array, os.path.join(file_path, filename_prefix, filename_prefix+'_preview.png'), mode=2) # create the preview image

    thumbnail_array = skimage.transform.resize(refl_array, (refl_array.shape[0] // 2, refl_array.shape[1] // 2),
                      anti_aliasing=False)
    # preview thumbnail
    print("Generating thumbnail image...")
    toRGB(thumbnail_array, os.path.join(file_path, filename_prefix, filename_prefix+'_thumbnail.png'), mode=2) # create the preview image


    # ----------------------------------------------------------------------------------------------------------------------------------------------

    # rename the metadata file
    os.rename(os.path.join(file_path, filename_prefix, filename_prefix+".json"),os.path.join(file_path, filename_prefix, filename_prefix+"_metadata.json"))
    print("Metadata file generated")


# ----------------------------------------------------------------------------------------------------------------------------------------

def pipeline(data_dir_path, output_data_path, desired_band_centres,
             desired_GSD = 5, nodata_thres = 0.5, output_mode = 'geotiff'
            ):
    """
    This script implements the image downsampling and rebanding pipeline.

    Parameters
    -----------
    data_dir_path : string / filepath-like
                Input image directory to find HDF5 files in.

    output_data_path : string / filepath-like
                Output image directory to write processed files in.

    desired_band_centres : 1-D array_like
                Array of the desired band centres to process input into.

    desired_GSD : integer, float
                Number in m units of desired ground sample distance to process into.

    nodata_thres : float
                Decimal fraction of missing data acceptable - images with more
                missing data then threshold will be skipped for processing

    --------
    Example Execution:
    --------
    def pipeline("/myinput/filepath/here", "/myoutput/filepath/here", [0.5, 0.586, 0.665, 0.804],
                desired_GSD = 5, nodata_thres = 0.5, output_mode = 'geotiff'
                )
    """
    downSampler_logger.info('-'*25+"New Run"+'-'*25) # logging
    downSampler_logger.info("pipeline function called: v_0.90") # logging

    ## set up our desired bands and GSD parameters, as well as our input and output files directory
    # 0.505, 0.526, 0.544, 0.565, 0.586, 0.606, 0.626, 0.646,  0.665, 0.682, 0.699, 0.715, 0.730, 0.745, 0.762, 0.779, 0.787, 0.804
    desired_band_centres = np.array(desired_band_centres)*1000 # convert to nm
    #desired_band_centres = np.array(desired_band_centres) # leave as um
    #desired_GSD = 4 # 4m GSD
    print("Parameters set...")

    # get all input HDF5 files
    file_dict = find_files(data_dir_path)
    print("Input files found...")

    for file_name, file_path in tqdm.tqdm(file_dict.items(), desc = "Processing image(s)"):
    #for file_name, file_path in file_dict.items():
        print("Processing file: ",file_name)
        # read the h5data2array for an image
        #file_name = 'NEON_D16_ABBY_DP3_552000_5071000_reflectance.h5'
        data_file_path = file_dict[file_name]
        refl_array, wavelength_array, FWHM_array, metadata_dict = h5data2array(data_file_path)
        print("File loaded...")

        # check missing data % and don't bother processing image if too high
        if np.round(np.count_nonzero(refl_array == metadata_dict['data ignore value'])/(refl_array.shape[0]*refl_array.shape[1]*refl_array.shape[2]),1) > nodata_thres:
           print("{} is missing > {} % threshold of data, processing skipped!".format(file_name, nodata_thres*100))
           continue

        print("Downsampling image...")
        # perform downsampling
        # params: img_array, GSD_input, GSD_output, input_bandcentres_array, output_bandcentres_array
        resamp_refl_array = downSample_reband_array(refl_array, metadata_dict['Spatial_Resolution_X_Y'][0], desired_GSD, wavelength_array, desired_band_centres) # downsample
        resamp_metadata_dict = metadata_dict.copy()
        resamp_metadata_dict['Spatial_Resolution_X_Y'] = [float(desired_GSD), float(desired_GSD)] # adjust resolution metadata to reflect downsampling
        rebanded_FWHM_array = np.round(band_widths(desired_band_centres),3)
        print("Downsampling complete!")



        if output_mode == 'hdf5':
            # generate image
            print("Generated image...")
            output_img_name = file_name.replace('NEON', 'Wyvern') # remove and replace NEON tag with Wyvern
            output_img_name = output_img_name.replace('.h5', 'preview') + '_img_' + str(desired_GSD) + 'mGSD.png' # remove .h5 ending and replace with img, GSD and .png
            output_data_path_filename = Path(output_data_path / 'preview_img' / output_img_name) # output path to save processed data files
            toRGB(resamp_refl_array, output_data_path_filename, mode=2) # generate and save image
            print("Image generated!")

            # write out file
            print(file_name,"Writting HDF5 file to disk...")
            output_hdf5_name = file_name.replace('NEON', 'Wyvern') # remove and replace NEON tag with Wyvern
            output_hdf5_name = output_hdf5_name.replace('.h5', '') + '_downsampled_' + str(desired_GSD) + 'mGSD.h5' # add downsampled, GSD and .h5 file ending
            output_data_path_filename = Path(output_data_path / 'hdf5_downsampled' / output_hdf5_name) # output path to save processed data files
            band_width_array = band_widths(desired_band_centres)
            array2h5data(resamp_refl_array, desired_band_centres, band_width_array, metadata_dict, output_data_path_filename) # save hdf5 data cube
            print(file_name,"HDF5 file written to disk!")

        elif output_mode == 'geotiff': # output geotiffs and STAC geojson metadata

            # set the time the image is created
            timestamp = datetime.datetime.utcnow()
            start_timestamp = timestamp + datetime.timedelta(0,-5)
            end_timestamp = timestamp + datetime.timedelta(0,5)

            time_string = str(timestamp).replace(' ','T') # add T in
            time_string = time_string.replace('-','') # remove hyphens
            time_string = time_string.replace(':','') # remove :
            time_string = time_string[0:15]

            # create filename based on company name, DS = Dragonette Simulated,
            # time string, 4 character NEON area code, and random 3 character string
            filename_prefix = 'WYVERN_DS_'+ time_string +'_'+ file_name[9:13] + rand_string(7).lower()
            downSampler_logger.debug("Generating filename prefix: "+filename_prefix)  # logging

            # Parent Directory path
            parent_dir = os.path.join(output_data_path, "catalog")
            # parent_dir = "../data/pipeline_output/catalog" # change this in production pipeline
            # make directory
            if not os.path.isdir(parent_dir):
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except OSError:
                    print(">>>>>> Creation of the directory {} failed".format(parent_dir))
                else:
                    print(">>>>>> Successfully created the directory {}".format(parent_dir))
                    #downSampler_logger.debug(">>>>>> Successfully created the directory {}".format(output_dir))  # logging

            # define some band names
            common_band_names = ['green1','green2','green3','green4','yellow',
                                 'red1','red2','red3','red4','red5','red6',
                                 'rededge1','rededge2','rededge3',
                                 'nir1','nir2','nir3','nir4'
                                ]

            # gnerate metadata file
            print('Generating STAC metadata file...')
            metadata2geojsonSTAC(resamp_refl_array, desired_band_centres,
                                rebanded_FWHM_array, resamp_metadata_dict,
                                common_band_names, timestamp, start_timestamp,
                                end_timestamp, parent_dir, filename_prefix
                                )
            # refl_array, metadata_dict, wavelength_array, FWHM_array, common_band_names, timestamp, start_timestamp, end_timestamp, file_path, filename_prefix="WYVERN_DS"
            print('STAC metadata file generated!')

            print('Generating geotiff image files...')
            # generate geotiff images
            output_dir = os.path.join(parent_dir, filename_prefix) # save geotiffs directly into the new directory that the STAC metadata file made
            array2gtiff_raster(resamp_refl_array, desired_band_centres,
                                rebanded_FWHM_array, resamp_metadata_dict,
                                resamp_metadata_dict['Spatial_Resolution_X_Y'][0], # pixel width
                                resamp_metadata_dict['Spatial_Resolution_X_Y'][1], # pixel height
                                output_dir, filename_prefix
                                )
            # refl_array, wavelength_array, FWHM_array, metadata_dict, pixelWidth, pixelHeight, file_path, filename_prefix="WYVERN_DS"
            print('Geotiff image files generated!')


        else:
            print('No file output mode selected')

    print("-" * 50)
    print("All Processing Completed!")
    downSampler_logger.info("pipeline processing complete!") # logging
    print("-" * 50)

# ----------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print("The value of __name__ is:", repr(__name__))
    #pipeline()
