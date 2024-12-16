"""
This program restores a series of jittered images with 
the jitter inferred in "infer_jitter_series.py".
"""
import sys
sys.path.append('./functions')
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import datetime as dt
import numpy as np
import math_func2 as mf
import pd_functions_v22 as pdf
import plots_func2 as pf
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import shift_func as sf
from tqdm import tqdm
plt.rcParams['figure.constrained_layout.use'] = True


"""
Imports and plots the set of Zernike coefficients and
the wavefront map over the different subfields.
"""
ind1=0#0 #First index of the series#
ind2=9#15#9 #Last index of the series
k_max=9 #Number of txt files employed to average the wavefront
low_f=0.2 #Cutoff of the Wiener filter
reg1=0.05 #0.02 #Regularization factor used when computing Q
reg2=1 #Regularization factor used when computing Q
cobs=32.4 #18.5 (MPS) 32.4 (Sunrise) #Diameter of central obscuration as a percentage of the aperture
Jmax=22# 16 or 22.Maximum index of the zernike polynomials
Jmin=4 #Minimum index of the zernike polynomials
#N0=400 #Size of the focused/defocused images in the FITS file
magnif=2.47 #Magnification factor of TuMag
plate_scale= 0.0378 #Plate scale in arcseconds (arcsec/pixel)
fps=3 #Number of frames per second for the movie

#Region to be subframed
crop=True #If True, it crops the image using x0, xf, y0 and yf as limits
x0=200 #200 or 400 #Initial pixel of the subframe in X direction
xf=x0+1600 #900 or 1600 #Final pixel of the subframe in X direction
y0=x0 #Initial pixel of the subframe in Y direction
yf=xf  #FInal pixel of the subframe in Y direction

#Path and name of the FITS file containing the focused and defocused images
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/Jitter/FS_1_2024_07_14'#'Flight/COMM_1_10_7_13_14' #Name of the folder containing th FITS file
pref='52502' #Prefilter employed
fname='series_jitter_2024_07_14'#'Icont_52502_cam_0_COMM_1' #Data file
ext='.fits' #Extention of the data file
txtfolder=dir_folder +'txt/PD_10_7_16_34_cam_0_52502_ima_1/svd' #Path of the txt files
fsigma='sigma_series_jitter_2024_07_14_wfe_corrected'#'sigma_'+fname
sigma=np.load(fsigma+'.npy')
cam=0 #0 or 1. Camera index
wave=0 #From 0 to 10. Wavelength index
modul=0 #From 0 to 4. Modulation index



"""
Read image
"""
if ext=='.npy':
    ima=np.load(dir_folder+ffolder+'/'+fname+ext)
    ima=ima[cam,wave,modul,:,:]
else:
    ima=pdf.read_image(dir_folder+ffolder+'/'+fname,ext,
                       norma='yes')


#Crop the image
if crop==True:
    if ima.ndim==2:
        ima=ima[x0:xf,y0:yf]
    elif ima.ndim==3:
        ima=ima[x0:xf,y0:yf,:]
    elif ima.ndim==4:
        ima=ima[:,0,x0:xf,y0:yf] #Select first modulation
        ima=np.moveaxis(ima,0,-1) #Move image index to last index
        ima=ima/np.mean(ima[:200,:200,0])#Normalize images to continuum



"""
Restoration with average Zernike coefficients if only one image is selected
"""
a_aver=pdf.retrieve_aberr(k_max,Jmax,txtfolder)


#########################
#a_aver=0*a_aver
#sigma[:,:]=0
###########################



a_d=0 #For pdf.object_estimate to restore a single image

#Image padding
ima_pad,pad_width=pdf.padding(ima)
cut=pad_width#To later select the non-padded region

   
"""
Correct from jitter if several images of the series is analyzed
"""
if ind2>ind1:
    ima_series=np.zeros((ima.shape[0],ima.shape[1],ind2-ind1+1))
    flag=np.zeros(ind2-ind1+1) #Zero if not corrected, one if already corrected
    j=-1
    for i in tqdm(range(ind1,ind2+1)):
        j+=1

        # Restore the image
        o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad[:,:,i],
                sigma[j,:],a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,
                reg1=reg1,reg2=reg2)
        ima_series[:,:,j]=o_plot[cut:-cut,cut:-cut] 
      
        if j==0:
            vmin=np.min(ima_series[:,:,j])
            vmax=np.max(ima_series[:,:,j])
    
     
    pf.movie3(ima,ima_series,'Icont_comparison.mp4',axis=2,fps=fps,
            title=['Jittered','Jitter free'])

#np.save(fname+'_wfe_corrected.npy',ima_series)
print('Sigma:')
print(np.round(sigma,3))

