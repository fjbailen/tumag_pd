"""
This program restores a Stokes cube using a given set of Zernike
coefficients and adding Gaussian noise with the same rms amplitud 
a given number of times. It restores individually each noisy Stokes
dataset and it averages all noisy datasets.
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
low_f=0.5 #Cutoff of the Wiener filter
reg1=0.1#0.05 #0.02 #Regularization factor used when computing Q
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
#Select cam, wavelength and Stokes component
cam=0
wave=-1
st=3 #V
st_lb=['I','Q','U','V']
ima0=demodulated_fft[cam,wave,st,:,:]
noise_rms=pdf.noise_rms(ima0)
print('Noise rms:',noise_rms)

#Compute noise filter with Stokes I
_,noise_filt=pdf.restore_ima(demodulated_fft[cam,-1,0,:,:],
                             zernikes,pd=0,low_f=low_f,noise='default',
                            reg1=reg1,reg2=reg2,cobs=cobs)      

#Restore Stokes component
ima_rest0,noise_filt=pdf.restore_ima(ima0,zernikes,pd=0,low_f=low_f,
                                    noise=noise_filt,
                                    reg1=reg1,reg2=reg2,cobs=cobs)      

vmin=np.min(ima_rest0)
vmax=np.max(ima_rest0)

#fig,axs=plt.subplots(1,2)
#im0=axs[0].imshow(ima,cmap='seismic',vmin=vmin,vmax=vmax)
#im1=axs[1].imshow(ima_rest,cmap='seismic',vmin=vmin,vmax=vmax)
#fig.colorbar(im0)
#fig.colorbar(im1)
#print(noise_rms)
#plt.show()
#quit()

#Add noise 50 times and average
"""
N=ima0.shape[0]
ima_rest=ima0*0
print('Adding noise a given number of realizations...')
for i in tqdm(range(100)):
    noisy_signal=np.random.normal(0,noise_rms,size=(N,N))
    ima=demodulated_fft[cam,wave,st,:,:] + noisy_signal

    ima_i,noise_filt=pdf.restore_ima(ima,zernikes,pd=0,low_f=low_f,
                                    noise=noise_filt,
                                    reg1=reg1,reg2=reg2,cobs=cobs)
    ima_rest+=ima_i
ima_rest=ima_rest/(i+1)


np.save('corrected_'+st_lb[st]+'_50_noisy_realizations.npy',ima_rest)           
"""
ima_rest=np.load('corrected_'+st_lb[st]+'_50_noisy_realizations.npy')

fig,axs=plt.subplots(1,2)
im0=axs[0].imshow(ima_rest0,cmap='seismic',vmin=vmin,vmax=vmax)
im1=axs[1].imshow(ima_rest,cmap='seismic',vmin=vmin,vmax=vmax)
fig.colorbar(im0)
fig.colorbar(im1)
plt.show()
quit()
###########


print('Restoring images from each cam...')
for cam in tqdm(range(2)):
    print('Restoring each wavelength')
    for wave in tqdm(range(Nwave)):
        for st in range(Nstokes):
            #If Stokes I: default noise filter
            if st==0:
                noise_filt='default'
            #Restore image and save it in array
            ima=demodulated_fft[cam,wave,st,:,:]
            ima_rest,noise_filt=pdf.restore_ima(ima,zernikes,pd=0,low_f=low_f,
                                                noise=noise_filt,
                                                reg1=reg1,reg2=reg2,cobs=cobs)
            demodulated_corrected[cam,wave,st,:,:]=ima_rest

        
            ####################################
            #Noise levels
            noise_rms=pdf.noise_rms(ima)
      
            #######################################


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


