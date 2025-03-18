"""
This program infers the residual crosstalk from I to Q, U and V
after demodulation of the intensity images and corrects for it
using the method described in Jaeggli et al. 2022.
(https://doi.org/10.3847/1538-4357/ac6506)

https://github.com/sajaeggli/adhoc_xtalk/blob/main/Jaeggli_etal_2022ApJ_AdHoc_Xtalk.ipynb

"""
import sys
sys.path.append('./functions')
import numpy as np
from matplotlib import colors, pyplot as plt
from astropy.io import fits
from glob import glob
from scipy.optimize import minimize
from scipy.ndimage import shift
from sklearn import linear_model
import pandas as pd
import jaeggli as jg
import pd_functions_v22 as pdf

# Read image
last_wvl=-1 #-1 or None. Last wavelength to be used in the fit
ext='.fits'#'.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/COMM_1_10_7_13_14' #Name of the folder containing th FITS file
#fname='stokes_COMM1_52502_demod_ffielded_avgflat'#'stokes_COMM1_52502_demod' #Data file
#fname='david_small_subregion_517'
fname='stokes_COMM1_517_demod_ffielded_avgflat'
path=dir_folder+ffolder+'/'+fname #Path of the image
I=fits.open(path+ext)
data = I[0].data
#data=np.load(path+'.npy')
print(data.shape)

data_corrected,MM1a=jg.fit_mueller_matrix(data,pthresh=0.02,norm=True,
                                          region=[200,1200,200,1200],
                                        last_wvl=last_wvl,plots=True)

print('Shape of restored data:',data_corrected.shape)
print('Diattenuation Mueller matrix:')
print(MM1a)


# Save the corrected data and the Mueller matrix
np.save('mueller_'+fname+'.npy',MM1a)
np.save('xtalk_corrected_'+fname+'.npy',data_corrected)
