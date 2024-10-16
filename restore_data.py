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
plt.rcParams['figure.constrained_layout.use'] = True


"""
Imports and plots the set of Zernike coefficients and
the wavefront map over the different subfields.
"""
ind1=0 #First index of the series
ind2=9 #Last index of the series
k_max=9 #Number of txt files employed to average the wavefront
low_f=0.2 #Cutoff of the Wiener filter
reg1=0.05 #0.02 #Regularization factor used when computing Q
reg2=1 #Regularization factor used when computing Q
cobs=32.4 #18.5 (MPS) 32.4 (Sunrise) #Diameter of central obscuration as a percentage of the aperture
Jmax=22# 16 or 22.Maximum index of the zernike polynomials
Jmin=4 #Minimum index of the zernike polynomials
#N0=400 #Size of the focused/defocused images in the FITS file
magnif=2.47 #Magnification factor of TuMag
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
ffolder='Flight/Jitter/FS_1_2024_07_14' #Name of the folder containing th FITS file
fname='series_jitter_2024_07_14' #Data file
ext='.fits' #Extention of the data file
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
else:
    ima=pdf.read_image(dir_folder+ffolder+'/'+fname,ext,
                       norma='yes',N=2016)
#Crop the image
if crop==True:
    if ima.ndim==2:
        ima=ima[x0:xf,y0:yf]
    elif ima.ndim==3:
        ima=ima[x0:xf,y0:yf,:]
    elif ima.ndim==4:
        ima=ima[:,0,x0:xf,y0:yf]
        ima=np.moveaxis(ima,0,-1) #Move image index to last index



"""
ima=ima[:,0,x0:xf,y0:yf]
contrast=100*np.std(ima[:,:500,:500],axis=(1,2))/np.mean(ima[:,:500,:500],axis=(1,2))
times=['14:11:02',
       '14:11:32',
       '14:12:01',
       '14:12:30',
       '14:13:00',
       '14:13:29',
       '14:13:59',
       '14:14:28',
       '14:14:58',
       '14:15:27']
#times = [dt.datetime.strptime(d,'%H:%M:%S') for d in times]
#pf.movie(ima,'series_jitter_2024_07_14.mp4',fps=3,axis=0,titles=times)
"""


ima_aligned=sf.realign_subpixel(ima,accu=0.05)
fig,axs=plt.subplots(1,2)
axs[0].imshow(ima[:,:,0]-ima[:,:,1],cmap='gray')
axs[1].imshow(ima[:,:,1]-ima_aligned[:,:,1],cmap='gray')
plt.show()
plt.close()
quit()


a_aver=pdf.retrieve_aberr(k_max,Jmax,cobs,txtfolder)

#########
a_aver=0*a_aver
##########


"""
Restoration with average Zernike coefficients
"""
a_d=0 #For pdf.object_estimate to restore a single image

#Image padding
ima_pad=pdf.padding(ima)
cut=int(0.1*ima_pad.shape[0]) #To later select the non-padded region

#If we select only one image of the series
if ind2==ind1: 
    ind=ind1 #Index of the image of the series to be restored
    o_plot,_,noise_filt=pdf.object_estimate(ima_pad[:,:,ind],a_aver,a_d,
                                            wind=True,cobs=cobs,cut=cut,
                                            low_f=low_f,reg1=reg1,reg2=reg2)
    o_plot=o_plot[cut:-cut,cut:-cut]
    contrast_0=np.std(ima[:,:,0])/(np.mean(ima[:,:,0]))*100
    contrast_rest=np.std(o_plot)/(np.mean(o_plot))*100
    min_rest=np.min(o_plot)
    max_rest=np.max(o_plot)



    """
    PLOTS
    """

    fig4,axs4=plt.subplots()
    axs4.errorbar(range(Jmin,Jmax),a_aver[(Jmin-1):]/(2*np.pi), yerr=a_rms[(Jmin-1):]/(2*np.pi),fmt='-o',capsize=3,color='k',label='Retrieved')
    axs4.set_ylim([-0.1,0.1])
    axs4.set_ylabel(r'Amplitude [$\lambda$]')
    axs4.set_xlabel('Zernike index')
    axs4.xaxis.set_minor_locator(MultipleLocator(1))
    axs4.xaxis.set_major_locator(MultipleLocator(5))


    #Plot original and restored images
    fig6,ax6=plt.subplots(1,2)
    plot1=ax6[0].imshow(ima[:,:,ind],cmap='gray',origin='lower',
                        interpolation='none',vmin=min_rest,vmax=max_rest)
    fig6.colorbar(plot1,ax=ax6[0])
    ax6[0].set_title('Contrast (%%): %.3g '%contrast_0)

    plot2=ax6[1].imshow(o_plot,cmap='gray',origin='lower',
                        interpolation='none',
                        vmin=min_rest,vmax=max_rest)
    fig6.colorbar(plot2,ax=ax6[1])
    ax6[1].set_title('Contrast (%%): %.3g '%contrast_rest)
    ax6[0].tick_params(axis='both', which='both', length=0)
    ax6[1].tick_params(axis='both', which='both', length=0)




    #Zoomed in images
    fig8,ax8=plt.subplots(1,2)
    plot1=ax8[0].imshow(ima[400:650,500:750,ind],cmap='gray',origin='lower',
                        interpolation='none',vmin=min_rest,vmax=max_rest)
    fig8.colorbar(plot1,ax=ax8[0])
    plot2=ax8[1].imshow(o_plot[400:650,500:750],cmap='gray',
                        origin='lower',vmin=min_rest,vmax=max_rest,
                        interpolation='none')
    fig8.colorbar(plot2,ax=ax8[1])
    ax8[0].tick_params(axis='both', which='both', length=0)
    ax8[1].tick_params(axis='both', which='both', length=0)


    """
    Auto-correlation to check residual aberrations
    """
    autocorr=mf.corr(ima[:,:,0],ima[:,:,0],norma=False)
    autocorr=autocorr/np.max(autocorr)
    autocorr_rest=mf.corr(o_plot,o_plot,norma=False)
    autocorr_rest=autocorr_rest/np.max(autocorr_rest)

    N2=int(ima.shape[0]/2)
    dx=50
    fig20,ax20=plt.subplots(1,2)
    ax20[0].imshow(np.abs(autocorr[(N2-dx):(N2+dx),(N2-dx):(N2+dx)]),cmap='gray')
    ax20[1].imshow(np.abs(autocorr_rest[(N2-dx):(N2+dx),(N2-dx):(N2+dx)]),cmap='gray')
    ax20[0].set_title('Focused')
    ax20[1].set_title('Restored')

    #Filter
    fig7,ax7=plt.subplots()
    ax7.imshow(noise_filt)

    #Radial power
    fig9,axs9=plt.subplots()
    power_radial_recons=pdf.power_radial(o_plot)
    power_radial_of0=pdf.power_radial(ima[:,:,0])
    nuc,_=pdf.compute_nuc(o_plot.shape[0])
    radius=np.arange(power_radial_recons.shape[0])/nuc
    axs9.semilogy(radius,power_radial_of0,label='Original')
    axs9.semilogy(radius,power_radial_recons,label='Restored')
    axs9.set_title('Radial power')
    axs9.legend()

    #plt.show()
    plt.close()
    quit()

#Movie of the whole series 
elif ind2>ind1:
    ima_series=np.zeros((ima.shape[0],ima.shape[1],ind2-ind1+1))
    flag=np.zeros(ind2-ind1+1) #Zero if not corrected, one if already corrected
    j=-1
    for i in range(ind1,ind2):
        j+=1
        print('-----------------')
        print('Image index %g'%i)
        print('-----------------')

        #Select the pair of images. The image with highest contras is selected as jitter-free
        contrast1=np.std(ima[:,:,i])/np.mean(ima[:,:,i])
        contrast2=np.std(ima[:,:,i+1])/np.mean(ima[:,:,i+1])
        if contrast1>contrast2:
            ima_jitt=ima[:,:,(i,i+1)]
            flag[j+1]+=1 #Index i+1 is corrected from jitter
        elif contrast2>contrast1:
            ima_jitt=ima[:,:,(i+1,i)]
            flag[j]+=1 #Index i is corrected from jitter
        
          
        #Realign the chosen pair of images
        kappa=20
        F0=fft2(ima_jitt[:,:,0])
        print('Re-aligning pair of images (%g,%g)'%(i,i+1))  
        F_comp=fft2(ima_jitt[:,:,1])
        error,row_shift,col_shift,Gshift=sf.dftreg(F0,F_comp,kappa)
        deltax=int(np.round(row_shift))
        deltay=int(np.round(col_shift))
        ima_jitt[:,:,1]=np.roll(ima_jitt[:,:,1],(deltax,deltay),axis=(0,1))
        print('Delta x, Delta y:',row_shift,col_shift)    
        
    
        #Compute jitter if not already corrected from it
        if flag[i]<2:
            Ok,gamma,wind,susf=pdf.prepare_PD(ima_jitt)
            sigma=pdf.minimization_jitter(Ok,cut=int(0.15*pdf.N))
            if contrast1>contrast2:
                print('Corrected image:',i+1)
            else:
                print('Corrected image:',i)
            print('Sigma',sigma)  
       
        
        # Restore the image
        if flag[j]==0:
            #Avoid jitter corrections for the image with index i
            o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad[:,:,i],
                [0,0,0],a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,reg1=reg1,reg2=reg2)
            ima_series[:,:,j]=o_plot[cut:-cut,cut:-cut]

            #Correct the  image with index i+1
            o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad[:,:,i+1],
                sigma,a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,reg1=reg1,reg2=reg2)
            ima_series[:,:,j+1]=o_plot[cut:-cut,cut:-cut] 
        elif flag[j]==1:
            #Correct the  image with index i
            o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad[:,:,i],
            sigma,a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,reg1=reg1,reg2=reg2)
            ima_series[:,:,j]=o_plot[cut:-cut,cut:-cut] 
   

        """
        sigma_vec=[[0.086,0.11,0.10],
                       [0,0,0],
                       [0.11,0.11,0.10],
                       [0,0,0],
                       [0.11,0.23,0],
                       [0,0.092,0],
                       [0,0,0],
                       [0,0.11,0.10],
                       [0.15,0.17,0],
                       [0.10,0.13,0.10],
                       [0,0,0],
                       [0.052,0,0.16],
                       [0,0,0],
                       [0.022,0.14,0],
                       [0.059,0,0],
                       [0,0,0]]
        sigma=sigma_vec[i]
        """




        if j==0:
            vmin=np.min(ima_series[:,:,j])
            vmax=np.max(ima_series[:,:,j])
    
    print('Flag',flag)
    #Restore last image of the series
    if flag[ind2]==0:
        #Correct the  image with index i
        o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad[:,:,ind2],
        [0,0,0],a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,reg1=reg1,reg2=reg2)
        ima_series[:,:,ind2]=o_plot[cut:-cut,cut:-cut] 

        
    #pf.movie(ima_series,'Icont_series_recons.mp4',axis=2,fps=5)
    pf.movie3(ima,ima_series,'Icont_comparison.mp4',axis=2,fps=fps,
            title=['Jittered','Jitter free'])



