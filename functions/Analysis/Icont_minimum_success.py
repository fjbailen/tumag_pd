
# ------------------------------ IMPORTS ----------------------------------------- #

import sys
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits

# Own Libs
sys.path.append("/home/users/dss/fbailen/TuMag/Pipeline/")
import config as cf
from utils import read_Tumag
from field_stop_finder import compute_alignment, apply_fieldstop_and_align_array
from master_dark import compute_master_darks
from master_flatfield import compute_master_flat_field
import image_handler as ih
from demodulation import demodulate
import phase_diversity as pd
np.seterr(divide='ignore')

obs_mod='D10-304-2739'
dark_indexes = "D10-5620-5719"
flats_index = "D10-4340-5619" #Mode 2.02


#Look for the images and separate observation modes
all_paths = ih.get_images_paths(obs_mod)
OCs = ih.separate_ocs(all_paths)
fe_ocs = [x for x in OCs if len(OCs[x]) == 64] #To select only the 2.02 mode (525.02 nm)
print(fe_ocs)

"""
#Compute darks
dark_paths = ih.get_images_paths(dark_indexes)
dc = compute_master_darks(dark_paths[:-6], verbose = True) # There is something strange in the last darks, you can see some structure -> Me los cargo con el :-6

# Compute master flat field: [cam, wavelength, stokes, dimx, dimy]
ff_obs_path = ih.get_images_paths(flats_index)
ff_obs, ff_obs_info = compute_master_flat_field(ff_obs_path, dc = dc, verbose = True)

# Select continuum wavelength and modulation 0 and normalize [cam, dimx, dimy]
ff = ff_obs[:, -1, 0] / np.max(ff_obs[:, -1, 0])
"""


#Read images and correct them
corr1=np.zeros((len(fe_ocs),2016,2016),dtype='float32')
corr2=np.zeros((len(fe_ocs),2016,2016),dtype='float32')
i=-1
for oc in fe_ocs:
    i+=1
    reordered_images = np.array(OCs[oc]).reshape(8, 4, 2)
    print(f"OC: {oc}:")
    print(f"----------")
    print(reordered_images[-1, 0])

    c1, H1 = ih.read(reordered_images[-1, 0, 0]) #Cam 0
    c2, H2 = ih.read(reordered_images[-1, 0, 1]) #Cam 1

    print(H1)
    quit()

    nacc = H2["nAcc"] #Get number of accumulations 

    #Correct from dark and flat fielding
    corr1[i,:,:] = (c1 - nacc*dc[0,:,:])/ff[0,:,:]
    corr2[i,:,:] = (c2 - nacc*dc[1,:,:])/ff[1,:,:]

#Save FITS [# of image, dimx, dimy]

hdu=fits.PrimaryHDU(corr1) #Cam 0
hdul=fits.HDUList([hdu])
hdul.writeto('Icont_cam_0_COMM_1.fits',overwrite=True) #Save cam 0 

hdu=fits.PrimaryHDU(corr2) #Cam 1
hdul=fits.HDUList([hdu])
hdul.writeto('Icont_cam_1_COMM_1.fits',overwrite=True) #Save cam 1

