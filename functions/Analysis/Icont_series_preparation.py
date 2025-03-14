
# ------------------------------ IMPORTS ----------------------------------------- #

import sys
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits

# Own Libs
sys.path.append("../")
import config as cf
from utils import read_Tumag
#from field_stop_finder import compute_alignment, apply_fieldstop_and_align_array
from master_dark import compute_master_darks
from master_flatfield import compute_master_flat_field
import image_handler as ih
from demodulation import demodulate
import phase_diversity as pd
np.seterr(divide='ignore')

obs_mod='D11-30968-41605' #2024-07-11 1h 50 min series
dark_indexes = "D11-41608-41707	"
flats_index = "D11-44410-46201" #Mode 2.02
Nwvl=1 #Number of wavelengths to be selected. Min:1. Max: 8 for the 2.02 mode.
cam=0 #Camera to be selected. 0 or 1

#Look for the images and separate observation modes
all_paths = ih.get_images_paths(obs_mod)
OCs = ih.separate_ocs(all_paths)


#To select only some observation counters
fe_ocs=np.arange(189,256,2)
#fe_ocs=np.arange(48,118) #For D14-45403-50000
#fe_ocs =np.arange(48,230) #For D14-45403-73480


#Compute darks
dark_paths = ih.get_images_paths(dark_indexes)
dc = compute_master_darks(dark_paths, verbose = True) # There is something strange in the last darks, you can see some structure -> Me los cargo con el :-6

# Compute master flat field correcting from dark: [cam, wavelength, modulation, dimx, dimy]
ff_obs_path = ih.get_images_paths(flats_index)
ff_obs, ff_obs_info = compute_master_flat_field(ff_obs_path, dc = dc, verbose = True)

#Read observations modes corrected from flat
corr1=np.zeros((len(fe_ocs),Nwvl,2016,2016),dtype='float32') #[image, wvl, dimx, dimy]
i=-1
hdr = fits.Header()
for oc in fe_ocs:
    i+=1
    print('OC:',oc)
    ob_mode = ih.nominal_observation("2.02", OCs[oc]["ims"],dc)
    data = ob_mode.get_data() #[cam, wvl, modulation, dimx, dimy]
  
    #We select cam and modulation 0
    for wvl in range(Nwvl):
        corr1[i,wvl,:,:]=data[cam,wvl,0,:,:]/ff_obs[cam,wvl,0,:,:]
    ob_mode_info = ob_mode.get_info()
    print(ob_mode_info['Images_headers']['wv_0']['M0']['image_name'])
    print(ob_mode_info['Images_headers']['wv_0']['M0']['Date'])
    hdr['fname_%g'%i] = ob_mode_info['Images_headers']['wv_0']['M0']['image_name']
    hdr['Time_%g'%i] = ob_mode_info['Images_headers']['wv_0']['M0']['Date'].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


#Change to float32
corr1=corr1.astype("float32") 

#Select a FOV from pixel 200 to 1600
corr1=corr1[:,:,200:1800,200:1800]


#Save FITS [# of image, dimx, dimy]
print(hdr)
hdu=fits.PrimaryHDU(corr1,header=hdr) #Cam 0
hdul=fits.HDUList([hdu])
hdul.writeto(obs_mod+'_cam'+str(cam)+'.fits',overwrite=True) #Save cam 0 





