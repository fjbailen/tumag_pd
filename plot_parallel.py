

import sys
sys.path.append('./functions')
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import math_func2 as mf
import pd_functions_v22 as pdf
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
plt.rcParams['figure.constrained_layout.use'] = True #For the layout to be as tight as possible

"""
Imports and plots the set of Zernike coefficients and
the wavefront map over the different subfields.
"""
N=300 #Dimension of the subpatches to run PD on
pref='517' #'517', '52502' or '52506'. Prefilter employed 
realign=False #Realign focused-defocused image with pixel accuracy?
Nima=1 #39 #Number of images in the series to be considered
cam=0 #Cam index: 0 or 1
low_f=0.2 #Noise filter threshold (default: 0.2)
reg1=0.05 #Regularization parameter for the restoration
reg2=1 #Regularization parameter for the restoration
cobs=32.4 #18.5 (MPS) 32.4 (Sunrise) #Diameter of central obscuration as a percentage of the aperture
Jmax=22# 16 or 22.Maximum index of the zernike polynomials
Jmin=4 #Minimum index of the zernike polynomials
wvl,fnum,Delta_x=pdf.tumag_params(pref=pref)
nuc,R=pdf.compute_nuc(N,wvl,fnum,Delta_x)
if wvl==517.3e-9:
    dmin=-1.51107 #Defocus in wavelength units 
elif wvl==525.02e-9 or wvl==525.06e-9:
    dmin=-1.489
dmax=dmin+0.05 #Maximum defocus in wavelength units
deltad=0.1 #Step of the defocus (>dmax-dmin if there is only one value of the defocus)
magnif=2.47 #Magnification factor of TuMag

#Region to be subframed
crop=True #If True, it crops the image using x0, xf, y0 and yf as limits
x0=400##Initial pixel of the subframe in X direction
xf=x0+900 #Final pixel of the subframe in X direction
y0=400 #Initial pixel of the subframe in Y direction
yf=y0+900  #FInal pixel of the subframe in Y direction

#Path and name of the FITS file containing the focused and defocused images
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/13_7_11_42' #Name of the folder containing th FITS file
fname='PD_13_7_11_42_cam_%g_%g_ima_%g'%(cam,int(pref),Nima) #Name of the FITS file
txtfolder=dir_folder +'txt'+ '/' + fname #Path of the txt files

#Colormap limits for wavefront representation
Npl=3 #Number of subplots in horizontal and vertical direction of main plot
vmin=-np.pi #Typically, pi/2 or pi
vmax=np.pi



"""
Read image
"""
ima=pdf.read_image(dir_folder+ffolder+'/'+fname,ext,norma='yes')

#Crop the image to use only a part of the pattern
if crop==True:
    if ima.ndim==4:
        ima=ima[:,:,x0:xf,y0:yf] #Select the region of interest
    elif ima.ndim==3:
        ima=ima[x0:xf,y0:yf,:]

#Re-order axis and normalize the image
if ima.ndim==4:
    ima=np.moveaxis(ima,0,-1)
    ima=ima[:Nima,:]#Select only one pair of images
    ima=np.flip(ima,axis=-1) #Invert the focused and defocused index
    ima=ima/np.mean(ima[:,:,0]) #Normalize by the mean of the focused image


#Realign
if realign is True:
    kappa=20
    for j in range(2):
        print('Re-aligning images with index %g'%j)
        F_focused=fft2(ima[0,:,:,j])
        for i in range(1,ima.shape[0]):
            F_comp=fft2(ima[i,:,:,j])
            error,row_shift,col_shift,Gshift=sf.dftreg(F_focused,F_comp,kappa)
            deltax=int(np.round(row_shift))
            deltay=int(np.round(col_shift))
            ima[i,:,:,j]=np.roll(ima[i,:,:,j],(deltax,deltay),axis=(0,1))
            print('Delta x, Delta y (pixels):',row_shift,col_shift)
elif realign is False:
    ima=np.mean(ima,axis=0) #Sum all images over the series


#Plot options
merit_plot=False
zernike_plot=True
multiple=True


"""
Loop
"""
cut=int(0.15*N)#29 #None#Subframing crop to avoid edges
ima_array=pdf.scanning(ima,Lsiz=N,cut=cut)
RHO,THETA=pdf.sampling2(N,R)
ap=pdf.aperture(N,R,cobs=cobs)

k_vec=np.arange(0,ima_array.shape[0])
k_max=k_vec.shape[0]


L_last=np.zeros(k_max)
rms_error=np.zeros(k_max)
norm_a_last=np.zeros(k_max)
it_v=np.zeros(k_max)
av=np.zeros((Jmax-1,k_max))
rms_labels=[]

if Npl>1:
    fig,axs=plt.subplots(Npl,Npl)
    fig2,axs2=plt.subplots(Npl,Npl)
    fig3,axs3=plt.subplots(Npl,Npl)
    fig5,ax5=plt.subplots()
    norm_a_array=np.zeros((Npl,Npl))


for k in k_vec:
    """
    Results after optimization files
    """
    filename=txtfolder+'/a_optimized_Jmax_%g_k_%g.txt'%(Jmax,k)
  
    #Import txt files
    data=np.genfromtxt(filename,delimiter='\t',unpack=False,dtype=None,\
    encoding='utf-8')
    names=np.array(data[:,0],dtype='str')
    values=data[:,1]

    #Obtain values from imported data
    a1_ind=np.argwhere(names=='a4')[0][0]
    a=np.array(values[a1_ind:],dtype='float64')
    a=np.concatenate((np.array([0,0,0]),a)) #First three are offset and tiptilt
    av[:,k]=a
    norm_a=2*np.pi/np.linalg.norm(a)
    maxit_ind=np.argwhere(names=='maxit')
    maxit=float(values[maxit_ind])
    wcut_ind=np.argwhere(names=='w_cut')
    w_cut=float(values[wcut_ind])

    #Wavefront
    wavef=pdf.wavefront(a,0,RHO,THETA,ap,R,N)


    """
    Plots
    """
    if Npl>1:
        #x and y indices
        n_vec,m_vec=np.unravel_index(k_vec,(Npl,Npl))
        n=n_vec[k]
        m=m_vec[k]

        norm_a_array[n,m]=norm_a

        #Zernike coefficients, original subframes and wavefront
        axs[n,m].plot(range(Jmin,Jmax),a[(Jmin-1):]/(2*np.pi),marker='.',label='k%.3g'%k,color='k')
        axs[n,m].set_ylim([-0.2,0.2])
        axs2[n,m].imshow(ima_array[k,:,:,0],cmap='gray')
        axs2[n,m].set_xticks([])
        axs2[n,m].set_yticks([])
        axs3[n,m].imshow(wavef,vmin=vmin,vmax=vmax,cmap='seismic')
        axs3[n,m].set_xticks([])
        axs3[n,m].set_yticks([])
        #axs3[n,m].set_title(r'$\lambda$/%.3g'%round(norm_a,2))

if Npl>1:
    #Optical quality over the image
    plot5=ax5.imshow(norm_a_array)
    ax5.set_title('RMS of wavefront in wave units')
    fig5.colorbar(plot5,ax=ax5)
    ax5.set_xticks([])
    ax5.set_yticks([])


#Find optimizations for which a=0 and delete them
index_ceros=np.argwhere(np.sum(np.abs(av), axis=0)==0)
av=np.delete(av, index_ceros, axis=1)

#2D plot of Zernike coefficients for each subpatch
fig18,ax18=plt.subplots()
ax18.imshow(av,cmap='seismic',vmin=-0.35,vmax=0.35)

#Average zernike coefficients and wavefront
a_aver=np.mean(av,axis=1)


a_rms=np.std(av,axis=1)
norm_aver=2*np.pi/np.linalg.norm(a_aver)
norm_aver2=2*np.pi/np.linalg.norm(a_aver[4:]) #To exclude tip/tilt and defocus
print('WFE RMS:lambda/',np.round(norm_aver,2))
wavef_aver=pdf.wavefront(a_aver,0,RHO,THETA,ap,R,N)
print(a_aver)

fig4,axs4=plt.subplots()
axs4.errorbar(range(Jmin,Jmax),a_aver[(Jmin-1):]/(2*np.pi), yerr=a_rms[(Jmin-1):]/(2*np.pi),fmt='-o',capsize=3,color='k',label='Retrieved')
axs4.set_ylim([-0.2,0.2])
axs4.set_ylabel(r'Amplitude [$\lambda$]')
axs4.set_xlabel('Zernike index')
axs4.xaxis.set_minor_locator(MultipleLocator(1))
axs4.xaxis.set_major_locator(MultipleLocator(5))

try:
    axs4.plot(range(Jmin,Jmax),a_aberr[(Jmin-1):(Jmax_ab-1)]/(2*np.pi),marker='o',label='Input')
    plt.legend()
except:
    print('WARNING: Input aberrations were not plotted')

#Average wavefront aberration
fig5,axs5=plt.subplots()
axs5.imshow(wavef_aver,vmin=vmin,vmax=vmax,cmap='seismic')
axs5.set_xticks([])
axs5.set_yticks([])
axs5.set_title(r'$\lambda$/%.3g'%round(norm_aver,2))



"""
Restoration with average Zernike coefficients
"""
#Select focused and defocused image before correction
Nrest=ima.shape[0]
of0=ima[:Nrest,:Nrest,0]
od0=ima[:Nrest,:Nrest,1]

#Defocuses
defoc=np.argwhere(names=='d')
try:
    defoc=float(values[defoc])
except ValueError:
    defoc=np.array([0,dmin])
a_d=defoc*np.pi/np.sqrt(3) #Defocus in radians


#Scene restoration
o_plot,noise_filt=pdf.restore_ima(ima,a_aver,pd=a_d,low_f=low_f,
                                  reg1=reg1,reg2=reg2,cobs=cobs)

contrast_0=np.std(of0)/(np.mean(of0))*100
contrast_rest=np.std(o_plot)/(np.mean(o_plot))*100
min_rest=np.min(o_plot)
max_rest=np.max(o_plot)


#Original and restored images
fig6,ax6=plt.subplots(1,2)
plot1=ax6[0].imshow(of0,cmap='gray',origin='lower',vmin=min_rest,vmax=max_rest)
fig6.colorbar(plot1,ax=ax6[0])
ax6[0].set_title('Contrast (%%): %.3g '%contrast_0)

plot2=ax6[1].imshow(o_plot,cmap='gray',origin='lower',vmin=min_rest,vmax=max_rest)

fig6.colorbar(plot2,ax=ax6[1])
ax6[1].set_title('Contrast (%%): %.3g '%contrast_rest)

ax6[0].tick_params(axis='both', which='both', length=0)
ax6[1].tick_params(axis='both', which='both', length=0)
#plt.savefig('./results/refocfine155_pair',transparent=True)
#plt.savefig(output + str(N) + 'px_restored.png')


"""
Auto-correlation to check residual aberrations
"""
if ima.ndim==3:
    autocorr=mf.corr(ima[:,:,0],ima[:,:,0],norma=False)
    autocorr=autocorr/np.max(autocorr)
    autocorr_rest=mf.corr(o_plot,o_plot,norma=False)
    autocorr_rest=autocorr_rest/np.max(autocorr_rest)

    N2=int(ima.shape[0]/2)
    dx=50
    fig20,ax20=plt.subplots(1,2)
    ax20[0].imshow(np.abs(autocorr[(N2-dx):(N2+dx),(N2-dx):(N2+dx)]),cmap='gray')
    ax20[1].imshow(np.abs(autocorr_rest[(N2-dx):(N2+dx),(N2-dx):(N2+dx)]),cmap='gray')
    ax20[0].set_title('Corr. focused')
    ax20[1].set_title('Corr. restored')




#Filter
fig7,ax7=plt.subplots()
ax7.imshow(noise_filt)
ax7.set_title('Noise filter')

#Radial power
fig9,axs9=plt.subplots()
power_radial_recons=pdf.power_radial(o_plot)
power_radial_of0=pdf.power_radial(of0)
nuc,_=pdf.compute_nuc(o_plot.shape[0],wvl,fnum,Delta_x)
radius=np.arange(power_radial_recons.shape[0])/nuc
axs9.semilogy(radius,power_radial_of0,label='Original')
axs9.semilogy(radius,power_radial_recons,label='Restored')
axs9.set_title('Radial power')
axs9.legend()

plt.show()
plt.close()


#Defocus inferred
defocus_length=a_aver[3]*(8*np.sqrt(3)*wvl*fnum**2)/(np.pi*magnif**2)
print('Defocus inferred at F4:',round(defocus_length*1e3,4),'mm')
print('Defocus inferred at image:',round(defocus_length*magnif**2*1e3,4),'mm')
quit()

fig8,ax8=plt.subplots()
ax8.imshow(o_plot,cmap='gray',vmin=min_rest,vmax=max_rest)
ax8.tick_params(axis='both', which='both', length=0)
plt.setp(ax8.get_xticklabels(), visible=False)
plt.setp(ax8.get_yticklabels(), visible=False)
#plt.savefig('./results/refocfine155_512px_corrected',dpi='original',transparent=True)
