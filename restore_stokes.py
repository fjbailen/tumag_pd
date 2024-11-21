"""
This program restores a Stokes cube using a given set of Zernike
coefficients.
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
from tqdm import tqdm
import shift_func as sf
plt.rcParams['figure.constrained_layout.use'] = True


"""
Imports and plots the set of Zernike coefficients and
the wavefront map over the different subfields.
"""
k_max=9 #Number of txt files employed to average the wavefront
low_f=0.2 #Cutoff of the Wiener filter
reg1=0.15#0.05 #0.02 #Regularization factor used when computing Q
reg2=1 #Regularization factor used when computing Q
cobs=32.4 #18.5 (MPS) 32.4 (Sunrise) #Diameter of central obscuration as a percentage of the aperture
Jmax=22# 16 or 22.Maximum index of the zernike polynomials
Jmin=4 #Minimum index of the zernike polynomials
magnif=2.47 #Magnification factor of TuMag
fps=3 #Number of frames per second for the movie

#Region to be subframed
crop=False #If True, it crops the image using x0, xf, y0 and yf as limits
x0=200 #200 or 400 #Initial pixel of the subframe in X direction
xf=x0+1600 #900 or 1600 #Final pixel of the subframe in X direction
y0=x0 #Initial pixel of the subframe in Y direction
yf=xf  #FInal pixel of the subframe in Y direction

#Path and name of the FITS file containing the focused and defocused images
ext='.npz' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/Demodulation' #Name of the folder containing th FITS file
fname='D10-304-2739_97_demod_cam' #Data file
txtfolder=dir_folder +'txt/PD_10_7_16_34_cam_0_52502_ima_1/svd' #Path of the txt files
cam=0 #0 or 1. Camera index
wave=0 #From 0 to 10. Wavelength index
modul=0 #From 0 to 4. Modulation index

#Plot options
merit_plot=False
zernike_plot=True
multiple=True


"""
Read image
"""
if ext=='.npy':
    ima=np.load(dir_folder+ffolder+'/'+fname+ext)
    ima=ima[cam,wave,modul,:,:]
elif ext=='.npz':
    dat = ffolder+'/'+fname+".npz"
    data = np.load(dat, allow_pickle=True)
    demodulated_fft = data['demodulated_fft'] #(cam,wave,stokes,Nx,Ny)
    data.close()    
    Nwave=demodulated_fft.shape[1]
    Nstokes=demodulated_fft.shape[2]
    Nx=demodulated_fft.shape[3]
    Ny=demodulated_fft.shape[4]

    #Normalize images
    for i in range(2):
        mu=np.mean(demodulated_fft[i,-1,0,:,:])
        demodulated_fft=demodulated_fft/mu   
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
        ima=ima[:,0,x0:xf,y0:yf]
        ima=np.moveaxis(ima,0,-1) #Move image index to last index

zernikes=pdf.retrieve_aberr(k_max,Jmax,txtfolder)


"""
Restoration with average Zernike coefficients
"""
demodulated_corrected=0*demodulated_fft


###########
noise_matrix=np.zeros((Nwave,Nstokes))
i=-1
for wave in tqdm(range(Nwave)):
        i+=1
        j=-1
        for st in range(Nstokes):
            j+=1
            ima=demodulated_fft[1,wave,st,:,:]
            noise_rms=pdf.noise_rms(ima)
            noise_matrix[i,j]=noise_rms

print(noise_matrix)
quit()
###########

noise_matrix=np.zeros((2,Nwave,Nstokes))
print('Restoring images from each cam...')
i=-1
for cam in tqdm(range(2)):
    i+=1
    print('Restoring each wavelength')
    j=-1
    for wave in tqdm(range(Nwave)):
        j+=1
        k=-1
        for st in range(Nstokes):
            k+=1
            #If Stokes I: default noise filter
            if st==0:
                noise_filt='default'
            #Restore image and save it in array
            ima=demodulated_fft[cam,wave,st,:,:]
            ima_rest,noise_filt=pdf.restore_ima(ima,zernikes,pd=0,low_f=low_f,
                                                noise=noise_filt,
                                                reg1=reg1,reg2=reg2,cobs=cobs)
            demodulated_corrected[cam,wave,st,:,:]=ima_rest

    
            #Noise levels of the original imae
            noise_rms=pdf.noise_rms(ima)
            noise_matrix[i,j,k]=noise_rms


            if st==1:
                vmin=-1200
                vmax=1200
                fig,axs=plt.subplots(1,2)
                axs[0].imshow(ima,cmap='seismic',vmin=vmin,vmax=vmax)
                axs[1].imshow(ima_rest,cmap='seismic',vmin=vmin,vmax=vmax)
                axs[0].set_title('%g, %g, %g'%(cam,wave,st))
                plt.show()
                quit()
            
           

np.savez(fname+'_corrected_reg1_%g'%reg1,'demodulated_fft',demodulated_corrected)           
quit()


