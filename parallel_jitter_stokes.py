"""
This program infers the residual jitter between two
images of the same scene using a TuMag's Stokes datacube. It uses
as jitter-free image the modulation with highest contrast for each
wavelength.
"""
import numpy as np
import os
import sys
sys.path.append('./functions')
import pd_functions_v22 as pdf
import multiprocessing as mtp
np.set_printoptions(precision=4,suppress=True)
from matplotlib import pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import shift_func as sf
import plots_func2 as pf
import math_func2 as mf
from tqdm import tqdm
plt.rcParams['figure.constrained_layout.use'] = True #For the layout to be as tight as possible


"""
Input parameters
"""
realign=False #Realign with pixel accuracy?
crop=False #If True, it crops the image using x0, xf, y0 and yf as limits
cobs=32.4 #Diameter of the central obscuration as a fraction of the primary mirror 
pref='517' #'517', '52502' or '52506'. Prefilter employed

#Region to be subframed
x0=200#400 #Initial pixel of the subframe in X direction
xf=x0+ 600#1600#900 #Final pixel of the subframe in X direction
y0=x0 #Initial pixel of the subframe in Y direction
yf=xf  #FInal pixel of the subframe in Y direction


#Path and name of the FITS file containing the focused and defocused images
dir_folder='./'
ffolder='Flight/Jitter/Crosstalk test'
fname='jitter_restored_cube'#'quadrant' #'corrected_magnesium'
output=fname+'_jitter/' #Name of output folder to save txt files
ext='.npy' #Extention of the data file
txtfolder=dir_folder +'txt/PD_13_7_11_42_cam_0_517_ima_1' #Path of the txt files
plate_scale=0.0378 #Plate scale in arcseconds (arcsec/pixel)

#Load image
#Data indices: cam, wavelength index, modulation, x, y
data=np.load(ffolder+'/'+fname+'.npy')


if crop is True:
    N=xf-x0
else:
    N=data.shape[-1] #Image dimensions

wvl,fnum,Delta_x=pdf.tumag_params(pref=pref)
nuc,R=pdf.compute_nuc(N,wvl,fnum,Delta_x)
cut=int(0.1*N) #29 #None#Subframing crop to avoid edges


"""
Load Stokes cube
"""
#Plot contrast for each camera, modulation and wavelength
contrast_matrix=np.zeros((2,4,10))
for i in range(2):
    for j in range(4):
        for k in range(10):
            dummy=data[i,k,j,:,:]
            contrast=np.std(dummy)/np.mean(dummy)*100
            contrast_matrix[i,j,k]=np.round(contrast,2)


fig,axs=plt.subplots(2,1)  
plot0=axs[0].imshow(contrast_matrix[0,:,:])
plot1=axs[1].imshow(contrast_matrix[1,:,:])
for i in range(2):
    axs[i].set_ylabel('Modulation')
    axs[i].set_xlabel('Wavelength index')
    axs[i].set_title('Cam %g'%i)
fig.colorbar(plot0)
fig.colorbar(plot1)

#Plot modulations for a given camera a wavelength
fig,axs=plt.subplots(1,4)
vmin=np.min(data[0,0,0,:,:])
vmax=np.max(data[0,0,0,:,:])
plot=axs[0].imshow(data[0,0,0,:,:]/np.mean(data[0,0,0,:,:]),cmap='gray')#,vmin=vmin,vmax=vmax)
fig.colorbar(plot)
axs[0].set_title('Modulation 1')
for i in range(1,4):
    plot=axs[i].imshow((data[0,0,i,:,:]-data[0,0,0,:,:])/np.mean(data[0,0,0,:,:]),cmap='seismic')#,vmin=vmin,vmax=vmax)
    fig.colorbar(plot)
    axs[i].set_title('Difference mod. %g'%(i+1))
plt.show()
quit()


#Correct all modulations for each cam and wavelength    
data_restored=0*data
for cam in tqdm(range(2)):
    for wvli in tqdm(range(10)):
    
        #Select the pair of jitter-free/jittered images
        ind1=np.argmax(contrast_matrix[cam,:,wvli])
        for ind2 in range(4):
            print('\n')
            print('-----------------------------')
            print('(cam,wvl,mod):',cam,wvli,ind2)
            ima=data[cam,wvli,(ind1,ind2),:,:]
            ima=np.moveaxis(ima,0,2) #Reorder for prepare_PD

     
            #Realign
            if realign is True:
                kappa=20
                F0=fft2(ima[:,:,0])
                for j in range(ima.shape[0]):
                    print('Re-aligning images with index %g'%j)  
                    F_comp=fft2(ima[:,:,j])
                    error,row_shift,col_shift,Gshift=sf.dftreg(F0,F_comp,kappa)
                    deltax=int(np.round(row_shift))
                    deltay=int(np.round(col_shift))
                    ima[j,:,:]=np.roll(ima[:,:,j],(deltax,deltay),axis=(0,1))
                    print('Delta x, Delta y:',row_shift,col_shift)
       
            """
            Jitter correction
            """
            Ok,gamma,wind,susf=pdf.prepare_PD(ima,nuc,N)

            #Find jitter if ind2 is not equal to ind1
            if ind1==ind2:
                sigma=[0,0,0]
            else:    
                sigma=pdf.minimization_jitter(Ok,gamma,plate_scale,nuc,N,cut=cut)
            print('Sigma:',sigma)
            zernikes=np.array([0,0,0,0])
            a_d=[0,0]
  
            #Pad image to reconstruct
            ima_pad,pad_width=pdf.padding(ima)

            #Reconstruction
            o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad,
                            sigma,zernikes,a_d,cobs=cobs,low_f=0.2,
                            wind=True,reg1=0.2,reg2=1)

            o_plot=o_plot[pad_width:-pad_width,pad_width:-pad_width]
            data_restored[cam,wvli,ind2,:,:]=o_plot

            print('Contrast original:',np.std(ima[:,:,1])/np.mean(ima[:,:,1]))
            print('Contrast restored:',np.std(o_plot)/np.mean(o_plot))
            
np.save('jitter_restored_cube.npy',data_restored)





