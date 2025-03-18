"""
This program is similar to infer_jitter_series, but uses
the function "correct_jitter_along_series" to process all
the data along the series.
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
from astropy.io import fits
plt.rcParams['figure.constrained_layout.use'] = True


"""
Imports and plots the set of Zernike coefficients and
the wavefront map over the different subfields.
"""
pref='52502' #'517', '52502' or '52506'. Prefilter employed
wave=0 #From 0 to 10. Wavelength index
modul=0 #From 0 to 3. Modulation index
cam=0 #0 or 1. Camera index
ind1=0 #First index of the series#
ind2=73 #180 (full series),70 (35 min seres) 15 (short series) #Last index of the series
k_max=9 #Number of txt files employed to average the wavefront
low_f=0.2 #Cutoff of the Wiener filter
reg1=0.05 #0.02 #Regularization factor used when computing Q
reg2=1 #Regularization factor used when computing Q
cobs=32.4 #18.5 (MPS) 32.4 (Sunrise) #Diameter of central obscuration as a percentage of the aperture
Jmax=22# 16 or 22.Maximum index of the zernike polynomials
Jmin=4 #Minimum index of the zernike polynomials
magnif=2.47 #Magnification factor of TuMag
plate_scale= 0.0378 #Plate scale in arcseconds (arcsec/pixel)
fps=3 #Number of frames per second for the movie

#Region to be subframed
crop=False #If True, it crops the image using x0, xf, y0 and yf as limits
x0=200 #200 #Initial pixel of the subframe in X direction
xf=x0+1600 #1600 #Final pixel of the subframe in X direction
y0=x0 #Initial pixel of the subframe in Y direction
yf=xf  #FInal pixel of the subframe in Y direction


#Path and name of the FITS file containing the focused and defocused images
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/QS8_11_7_9_05' #Name of the folder containing th FITS file
pref='517'#'52502' #Prefilter employed
fname='D11-30968-41605_cam%g_mod%g'%(cam,modul) #Data file
ext='.fits'#'.npy' #Extention of the data file
txtname='PD_11_7_11_00_cam_0_52502_ima_1'#'zeros' if no PD analysis is available
txtfolder=dir_folder +'txt/'+txtname #Path of the txt files
moviename='Icont_jitter_'+fname+'.mp4'#Name of the movie to be saved



"""
Read image and select region and frame interval
"""
path=dir_folder+ffolder+'/'+fname #Path of the image
I=fits.open(path+ext)
ima = I[0].data
print('FITS shape:',ima.shape)
if crop is True:
    ima=ima[ind1:ind2,0,x0:xf,y0:yf]
else:
    ima=ima[ind1:ind2,0,:,:]    
print('Imported data shape:',ima.shape)  


"""
Find image with highest contrast
"""
Nframes=ima.shape[0]
contrast=np.zeros(Nframes)
for i in range(Nframes):
    contrast[i]=np.std(ima[i,:,:])/np.mean(ima[i,:,:])

#Plot of contrast along the series
fig,axs=plt.subplots()
axs.plot(100*contrast,marker='o')
axs.set_xlabel('Image index')
axs.set_ylabel('Contrast (%)')

#Plot of the first image of the series
fig,axs=plt.subplots()
axs.imshow(ima[0,:,:],cmap='gray')
fig.colorbar
plt.show()


#Correct jitter
a_aver=pdf.retrieve_aberr(k_max,Jmax,txtfolder)
ima_corr,sigma_vec=pdf.correct_jitter_along_series(ima,a_aver,pref=pref,cut=None,
                                cobs=cobs,low_f=low_f,reg1=reg1,reg2=reg2,
                                plate_scale=plate_scale,print_res=False)

print(np.round(sigma_vec,3))
np.save('sigma_'+fname+'.npy',sigma_vec)

            
#Save movie
ima=np.moveaxis(ima,0,-1)
ima_corr=np.moveaxis(ima_corr,0,-1)
pf.movie3(ima[1:,:],ima_corr[1:,:],moviename,axis=2,fps=fps,
            title=['Original','WFE+jitter'])

quit()


#Restore images only from aberrations
ima_wfe=ima.copy()
for i in range(Nframes): 
    o_plot2,_,noise_filt2=pdf.object_estimate_jitter(ima_pad[:,:,i],
                    0*sigma_vec[i,:],a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,reg1=reg1,reg2=reg2)
    ima_wfe[:,:,i]=o_plot2[cut:-cut,cut:-cut]

pf.movie13(ima,ima_wfe,ima_series,'original_vs_wfe_jitter_corrected_mod%g.mp4'%modul,
           axis=2,fps=fps,title=['Original','WFE','WFE+jitter'])