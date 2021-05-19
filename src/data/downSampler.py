

# libraries
# data
import pandas as pd
import numpy as np
import h5py
from PIL import Image

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

# dataframe options
pd.set_option('display.max_columns', None)

#%matplotlib inline

#import warnings
#warnings.filterwarnings('ignore')

# error logging


import logging
downSampler_logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------------------

def find_files(data_dir_path):
    """
    This function takes in a path to a data directory, walks the directory and
    returns a dictionary of the filenames and paths of all the data files.
    """
    downSampler_logger.debug("find_files function called")  # logging

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
    Iterate through groups in a HDF5 file and prints the groups and datasets
    names and datasets attributes
    """
    downSampler_logger.debug("descend_obj function called") # logging

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
    refl_clean, wavelength_array, FWHM_array, metadata = h5data2array("NEON_D16_ABBY_DP3_552000_5071000_reflectance.h5")

    """
    downSampler_logger.debug("h5data2array function called") # logging


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
    downSampler_logger.debug("array2h5data function called") # logging


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


def reband_spectral_array(spect_array, input_bandcentres_array, output_bandcentres_array):
    """
    Takes in both a 1D spectral array and a input array of band centres and then returns the linearly interpolated
    reflectance values of the desired output bands.
    NOTE: input band centre wavelengths are all in nm units

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
    #downSampler_logger.debug("reband_spectral_array function called") # logging


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
    downSampler_logger.debug("band_widths function called") # logging

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
    downSampler_logger.debug("downSample_reband_array function called") # logging

    downsample_factor = max(int(round(GSD_output/GSD_input)),1) # calculate the downsampling factor - only integers!
    rescale_factor = float(downsample_factor)
    downSampler_logger.debug("GSD_output passed: {}, GSD_input passed: {}, rescale_factor calculated: {}".format(GSD_output, GSD_input,rescale_factor)) # logging

    img_array_ds = skimage.measure.block_reduce(img_array[:, :, :],
                                             block_size=(downsample_factor, downsample_factor, 1),
                                             func=np.mean) # downsample image array
    downSampler_logger.debug("img_array shape passed: {}, img_array_ds shape downsampled: {}".format(img_array.shape, img_array_ds.shape)) # logging
    downSampler_logger.debug("img_array: {}, img_array_ds downsampled: {}".format(img_array, img_array_ds)) # logging

    # create empty array to store the rebanded spectral arrays
    rebanded_array = np.zeros((img_array_ds.shape[0], img_array_ds.shape[1], len(output_bandcentres_array)))
    downSampler_logger.debug("Should be empty - rebanded_array shape: {}".format(rebanded_array.shape)) # logging
    downSampler_logger.debug("Should be empty - rebanded_array: {}".format(rebanded_array)) # logging


    # loop through all spatial pixels and reband the spectral component of each one
    for x in range(0,img_array_ds.shape[0]): # x dimension
        for y in range(0,img_array_ds.shape[1]): # y dimension

            downSampler_logger.debug("Passed to reband_spectral_array - img_array_ds[x, y, :]: {}, \ninput_bandcentres_array: {}, \noutput_bandcentres_array: {}".format(img_array_ds[x, y, :], input_bandcentres_array, output_bandcentres_array)) # logging
            rebanded_array[x, y, :] = np.round_(reband_spectral_array(img_array_ds[x, y, :],
                                                            input_bandcentres_array,
                                                            output_bandcentres_array),4) # build up empty array with rebanded arrays

    downSampler_logger.debug("Should be full now - rebanded_array shape: {}, rebanded_array: {}".format(rebanded_array.shape, rebanded_array)) # logging
    # upscale image back to original input image size (interpolating pixels in between)
    downSampler_logger.debug("Rescale of the rebanded_array shape: {}, rescaled rebanded_array: {}".format(skimage.transform.rescale(rebanded_array,(rescale_factor,rescale_factor,1.0)).shape, skimage.transform.rescale(rebanded_array,(rescale_factor,rescale_factor,1.0)))) # logging
    return skimage.transform.rescale(rebanded_array,(rescale_factor,rescale_factor,1.0)) # scale back up to original size (interpolates)


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

def pipeline(data_dir_path, output_data_path, desired_GSD = 4,
                desired_band_centres = [0.505, 0.526, 0.544, 0.565, 0.586, 0.606, 0.626, 0.646, 0.665, 0.682, 0.699, 0.715, 0.730, 0.745, 0.762, 0.779, 0.787, 0.804],
            ):
    """
    This script implements the image downsampling and rebanding pipeline.

    """
    # print("This is my file to test Python's execution methods.")
    # print("The variable __name__ tells me which context this file is running in.")
    # print("The value of __name__ is:", repr(__name__))


    ## set up our desired bands and GSD parameters, as well as our input and output files directory
    # 0.505, 0.526, 0.544, 0.565, 0.586, 0.606, 0.626, 0.646,  0.665, 0.682, 0.699, 0.715, 0.730, 0.745, 0.762, 0.779, 0.787, 0.804
    #desired_band_centres = np.array([0.505, 0.526, 0.544, 0.565, 0.586, 0.606, 0.626, 0.646, 0.665, 0.682, 0.699, 0.715, 0.730, 0.745, 0.762, 0.779, 0.787, 0.804])
    desired_band_centres = np.array(desired_band_centres)*1000 # convert to nm
    #desired_GSD = 4 # 4m GSD
    #data_dir_path = Path(os.getcwd()).parents[0] / 'data' / 'NEON' # get path to data files
    #output_data_path = Path(os.getcwd()).parents[0] / 'data' / 'interim' # output data directory
    print("Parameters set...")

    # get all input HDF5 files
    file_dict = find_files(data_dir_path)
    print("Input files found...")

    for file_name, file_path in file_dict.items():
        print("Processing file: ",file_name)
        # read the h5data2array for an image
        #file_name = 'NEON_D16_ABBY_DP3_552000_5071000_reflectance.h5'
        data_file_path = file_dict[file_name]
        refl_array, wavelength_array, FWHM_array, metadata_dict = h5data2array(data_file_path)
        print("File loaded...")

        # perform downsampling
        resamp_refl_array = downSample_reband_array(refl_array, metadata_dict['Spatial_Resolution_X_Y'][0], desired_GSD, wavelength_array, desired_band_centres) # downsample
        metadata_dict['Spatial_Resolution_X_Y'] = [float(desired_GSD), float(desired_GSD)] # adjust resolution metadata to reflect downsampling
        print("Downsampling Complete...")

        # generate image
        output_img_name = file_name.replace('NEON', 'Wyvern') # remove and replace NEON tag with Wyvern
        output_img_name = output_img_name.replace('.h5', 'preview') + '_img_' + str(desired_GSD) + 'mGSD.png' # remove .h5 ending and replace with img, GSD and .png
        output_data_path_filename = Path(output_data_path / 'preview_img' / output_img_name) # output path to save processed data files
        toRGB(resamp_refl_array, output_data_path_filename, mode=2) # generate and save image
        print("Image Generated...")

        # write out file
        output_hdf5_name = file_name.replace('NEON', 'Wyvern') # remove and replace NEON tag with Wyvern
        output_hdf5_name = output_hdf5_name.replace('.h5', '') + '_downsampled_' + str(desired_GSD) + 'mGSD.h5' # add downsampled, GSD and .h5 file ending
        output_data_path_filename = Path(output_data_path / 'hdf5_downsampled' / output_hdf5_name) # output path to save processed data files
        band_width_array = band_widths(desired_band_centres)
        array2h5data(resamp_refl_array, desired_band_centres, band_width_array, metadata_dict, output_data_path_filename) # save hdf5 data cube
        print(file_name,"HDF5 File Written to Disk...")

    print("-" * 50)
    print("All Processing Completed!")
    print("-" * 50)

# ----------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    pipeline()
