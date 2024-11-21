"""
This program infers the residual jitter between two
images of the same scene.
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



"""
Input parameters
"""
check_image=False
realign=True
crop=True #If True, it crops the image using x0, xf, y0 and yf as limits
cobs=32.4 #Diameter of the central obscuration as a fraction of the primary mirror 
pref='52502' #'517', '52502' or '52506'. Prefilter employed

#Region to be subframed
ind1=7 #Image free from jitter
ind2=8 #Image affected by jitter
x0=200#400 #Initial pixel of the subframe in X direction
xf=x0+ 600#1600#900 #Final pixel of the subframe in X direction
y0=x0 #Initial pixel of the subframe in Y direction
yf=xf  #FInal pixel of the subframe in Y direction
N=xf-x0 #Dimension of the image

#Path and name of the FITS file containing the focused and defocused images
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/Jitter/FS_1_2024_07_14' #Name of the folder containing th FITS file
fname='series_jitter_2024_07_14' #Data file
output=fname+'_jitter/' #Name of output folder to save txt files
ext='.fits' #Extention of the data file
txtfolder=dir_folder +'txt/PD_10_7_16_34_cam_0_52502_ima_1/svd' #Path of the txt files
cam=0 #0 or 1. Camera index
wave=0 #From 0 to 10. Wavelength index
modul=0 #From 0 to 4. Modulation index
plate_scale=0.0378 #Plate scale in arcseconds (arcsec/pixel)
wvl,fnum,Delta_x=pdf.tumag_params(pref=pref)
nuc,R=pdf.compute_nuc(N,wvl,fnum,Delta_x)
cut=int(0.1*N) #29 #None#Subframing crop to avoid edges


"""
Read image
"""
if ext=='.npy':
    ima=np.load(dir_folder+ffolder+'/'+fname+ext)
    ima=ima[cam,wave,modul,:,:]
else:
    ima=pdf.read_image(dir_folder+ffolder+'/'+fname,ext,
                       norma='yes')

if crop==True:
    if ima.ndim==2:
        ima=ima[x0:xf,y0:yf]
    elif ima.ndim==4:
        ima=ima[:,0,x0:xf,y0:yf]
        ima=np.moveaxis(ima,0,-1) #Move image index to last index

#Select the image with highest contrast as the jitter-free one
contrast1=np.std(ima[:,:,ind1])/np.mean(ima[:,:,ind1])
contrast2=np.std(ima[:,:,ind2])/np.mean(ima[:,:,ind2])
if contrast1>contrast2:
    ima=ima[:,:,(ind1,ind2)]
elif contrast2>contrast1:
    ima=ima[:,:,(ind2,ind1)]    

#Realign
if realign is True:
    kappa=20
    F0=fft2(ima[:,:,0])
    for j in range(ima.shape[-1]):
        print('Re-aligning images with index %g'%j)  
        F_comp=fft2(ima[:,:,j])
        error,row_shift,col_shift,Gshift=sf.dftreg(F0,F_comp,kappa)
        deltax=int(np.round(row_shift))
        deltay=int(np.round(col_shift))
        ima[:,:,j]=np.roll(ima[:,:,j],(deltax,deltay),axis=(0,1))
        print('Delta x, Delta y:',row_shift,col_shift)



if check_image is True:
    fig,axs=plt.subplots(1,2,layout="constrained")
    if ima.ndim==4:
        ima_foc=ima[0,:,:,0]
        ima_defoc=ima[0,:,:,1]

    elif ima.ndim==3:
        ima_foc=ima[:,:,0]
        ima_defoc=ima[:,:,1]  
    contrast_foc=100*np.std(ima_foc)/np.mean(ima_foc)
    contrast_defoc=100*np.std(ima_defoc)/np.mean(ima_defoc) 
    print('Contrast im 0:',np.round(contrast_foc,1),'%')
    print('Contrast im 1:',np.round(contrast_defoc,1),'%')

    axs[0].imshow(ima_foc,cmap='gray',interpolation='none')
    axs[1].imshow(ima_defoc,cmap='gray',interpolation='none')
    axs[0].set_title('Jitter-free')
    axs[1].set_title('Affected by jitter')
    plt.show()
    quit()




"""
Preparation of PD
"""

Ok,gamma,wind,susf=pdf.prepare_PD(ima,nuc,N)
sigma=pdf.minimization_jitter(Ok,gamma,plate_scale,nuc,N,cut=cut)
print('Sigma:',sigma)
a=np.array([0,0,0,0])
a_d=[0,0]


#Pad image to reconstruct
ima_pad,pad_width=pdf.padding(ima)

#Reconstruction
o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad,
                sigma,a,a_d,cobs=cobs,low_f=0.2,
                wind=True,reg1=0.2,reg2=1)


o_plot=o_plot[pad_width:-pad_width,pad_width:-pad_width]
contrast_rest=np.round(100*np.std(o_plot)/np.mean(o_plot),1)
contrast0=np.round(100*np.std(ima[:,:,1])/np.mean(ima[:,:,1]),1)

print('Contrast (%):',contrast_rest)
vmin=np.min(o_plot)
vmax=np.max(o_plot)

fig, axs= plt.subplots(1,2,layout='constrained')
axs[0].imshow(ima[:,:,1],cmap='gray',vmin=vmin,vmax=vmax)
axs[1].imshow(o_plot,cmap='gray',vmin=vmin,vmax=vmax)
axs[0].set_title(r'Affected by jitter (%g'%contrast0 + r'$\%$)')
axs[1].set_title(r'Corrected from jitter (%g'%contrast_rest + r'$\%$)')
#plt.colorbar()
plt.show()
quit()
############



try:
    os.mkdir('./txt/'+output)
except FileExistsError:
    print('./txt/'+output+' already created')

def subpatch(k):
    guess='False' #No preparado para usar como guess la solución de otra caja

    """
    PD
    """
    #Outputs of prepare_D
    if ima_array.shape[0]==0:#If no subfielding
        Ok,gamma,wind,susf=pdf.prepare_PD(ima)
    else: #If subfielding (pdf.N < data X/Y dimensions)
        Ok,gamma,wind,susf=pdf.prepare_PD(ima_array[k,:,:,:])

    #Call to optimization function
    if optimization=='linear':
        a=pdf.loop_opt(tol,Jmin,Jmax,w_cut,maxnorm,maxit,\
        a0,a_d,RHO,THETA,ap,Ok,cut=cut,method=svd_meth,gamma=gamma,K=K)  
        
    elif optimization=='lbfgs':
        a=pdf.minimization(Jmin,Jmax,a0,a_d,RHO,THETA,ap,Ok,\
        cut=cut,gamma=gamma,K=2,jac=True)

 
    """
    Save txt and images
    """
    #Aberrations and parameters txt file
    flabel=['file','output','ext','Jmin','Jmax',\
    'tol','maxnorm','maxit','w_cut','d','a_d']
    ll=-1
    if Jmin==2: #Save tip/tilt terms
        for i in range(len(a)):
            if i>(2*K):
                flabel.append('a%g'%(i+3-2*K))
            elif i>0:
                if i%2==1:
                    ll+=1
                    flabel.append('tiltx%g'%ll)
                else:
                    flabel.append('tilty%g'%ll)
            elif i==0:
                flabel.append('offset')
    else:
        for i in range(len(a)):
            flabel.append('a%g'%(i+1))
    param=np.array([fname,ffolder,ext,Jmin,Jmax,\
    tol,maxnorm,maxit,w_cut,a_d*np.sqrt(3)/np.pi,a_d],dtype=object)
    filea=np.concatenate((param.reshape(len(param),1),a))
    filea=np.column_stack((flabel,filea))
    filename='./txt/'+output+'a_optimized_Jmax_%g_k_%g.txt'%(Jmax,k)
    np.savetxt(filename,filea,delimiter='\t',fmt='%s',encoding='utf-8')

    #Process information
    proc_name = mtp.current_process().name
    print('Process', proc_name)


#Parallel computing!!!!
if __name__ == '__main__':
    p=mtp.Pool(n_cores)
    p.map(subpatch,k_vec,chunksize=1)
