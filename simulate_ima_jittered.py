"""
This function reads an MHD simulation and simulates the presence of jitter
of a given amplitude.
"""
from PIL import Image as im
import numpy as np
import sys
sys.path.append('./functions')
import pd_functions_v22 as pdf
np.set_printoptions(precision=4,suppress=True)
from matplotlib import pyplot as plt
import general_func as gf
from scipy.fftpack import fft2,ifft2
import shift_func as sf
import plots_func2 as pf
from tqdm import tqdm
plt.rcParams['figure.constrained_layout.use'] = True #For the layout to be as tight as possible
plt.rcParams["image.interpolation"] = 'none'


#Parameters of input data
file='cadence' #MHD simulation to be opened. 'continuo' or 'cadence'
SNR=0 #Signal-to-noise ratio. 0 if no noise is to be applied
cobs=32.4 #Diameter of central obscuration as a percentage of the aperture
plate_scale=0.055 #Plate scale of the simulations in arcseconds (arcsec/pixel)
sigmax=0.10 #RMS of jitter along X direction (px)
sigmay=0.10 #RMS of jitter along X direction (px)
Nacc=1000 #Number of accumulated images
pref='52502' #'517', '52502' or '52506'. Prefilter employed
aberr=0.0001*np.ones(6) #Aberrations of the instrument



"""
Read image
"""
if file=='continuo':
    N=288 #Number of pixels of the image 
    fname='./continuo' #MHD simulation
    ext='.sav' #MHD simulation
    ima0=pdf.read_image(fname,ext)
    ima0=ima0[:N,:N]
elif file=='cadence':
    N=256 #Number of pixels of the image
    fname='./map_1s-cadence' #Cadence simulation
    ext='.npz' #Cadence simulation
    ima0=pdf.read_image(fname,ext)
    ima0=ima0[:N,:N,0]
output='./results/'
ima0=ima0/np.mean(ima0[:256,:256]) #Normalize only by QS mean intensity
ima0=ima0.astype('float64')

#System parameters
wvl,fnum,Delta_x=pdf.tumag_params(pref=pref)
nuc,R=pdf.compute_nuc(N,wvl,fnum,Delta_x)
RHO,THETA=pdf.sampling2(N,R) 
ap=pdf.aperture(N,R,cobs=cobs)


"""
Simulate the image affected by jitter with sigmax and sigmay
"""
ima_shift,rms_x,rms_y=pdf.simulate_jitter(ima0,sigmax,sigmay,
                                          plate_scale,Nacc)


"""
Apply the telescope diffraction and add noise
"""
ima=pdf.convPSF(ima0,aberr,0,RHO,THETA,ap,norm=True)
ima_shift=pdf.convPSF(ima_shift,aberr,0,RHO,THETA,ap,norm=True)
if SNR>0:
    NSR=1/SNR
    ima=pdf.gaussian_noise(NSR,ima)
    ima_shift=pdf.gaussian_noise(NSR,ima_shift)


"""
Save images in FITS file
"""
ima_array=np.zeros((N,N,2))
ima_array[:,:,0]=ima
ima_array[:,:,1]=ima_shift
ima_name='mhd_jitter_sigmax_%g_sigmay_%g'%(sigmax,sigmay)
#pdf.save_image(ima_array,ima_name,folder='')


"""
Infer the jitter and restore the image
"""
cut=int(0.1*N)
Ok,gamma,wind,susf=pdf.prepare_PD(ima_array,nuc,N)
sigma=pdf.minimization_jitter(Ok,gamma,plate_scale,nuc,N,cut=cut)
print('Sigma (arcsec):\n',sigma)
#print('Sigma (px units):',sigma/plate_scale)
print('True sigma (arcsec):',rms_x,rms_y)



#Restore jittered image
ima_pad,pad_width=pdf.padding(ima_shift)
o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad,
                sigma,aberr,0,cobs=cobs,low_f=0.2,
                wind=True,reg1=0.05,reg2=1,inst='imax')
o_plot=o_plot[pad_width:-pad_width,pad_width:-pad_width]


#Restore jittered-free image
ima_pad_nojitter,pad_width=pdf.padding(ima)
o_plot_nojitter,_,noise_filt=pdf.object_estimate_jitter(ima_pad,
                0*sigma,aberr,0,cobs=cobs,low_f=0.2,
                wind=True,reg1=0.05,reg2=1,inst='imax')
o_plot_nojitter=o_plot_nojitter[pad_width:-pad_width,pad_width:-pad_width]


"""
Compute contrasts and plot
"""

#Plots

fig,axs=plt.subplots(1,4)
axs[0].imshow(ima0,cmap='gray') #Jitter-free original MHD image
axs[0].set_title('MHD simulation')
axs[1].imshow(ima,cmap='gray') #Jitter-free image after diffraction and noise
axs[1].set_title('Diffraction')
axs[2].imshow(ima_shift,cmap='gray') #Jittered image after diffraction and noise
axs[2].set_title('Diffraction + Jitter')
axs[3].imshow(o_plot,cmap='gray') #Restored image
axs[3].set_title('Reconstructed')
pf.remove_tick_labels(axs)

fig,axs=plt.subplots(2,2)
axs[0,0].imshow(ima,cmap='gray') #Jitter-free image after diffraction and noise
axs[0,0].set_title('Jitter-free',fontsize=11)
axs[0,0].set_ylabel('Original',fontsize=11)
axs[0,1].imshow(ima_shift,cmap='gray') #Jittered image after diffraction and noise
axs[0,1].set_title('Jittered',fontsize=11)
axs[1,0].imshow(o_plot_nojitter,cmap='gray') #Jitter-free restored image
axs[1,0].set_ylabel('Restored',fontsize=11)
axs[1,1].imshow(o_plot,cmap='gray') #Restored image
pf.remove_tick_labels(axs)

#Contrasts
cima0=np.round(100*np.std(ima0)/np.mean(ima0),1)
cima=np.round(100*np.std(ima)/np.mean(ima),1)
cima_shift=np.round(100*np.std(ima_shift)/np.mean(ima_shift),1)
c_rest=np.round(100*np.std(o_plot)/np.mean(o_plot),1)
c_rest_nojitter=np.round(100*np.std(o_plot_nojitter)/np.mean(o_plot_nojitter),1)
contrast_array=np.array([cima,cima_shift,c_rest_nojitter,c_rest])

#Add text labels with the contrast values
k=-1
for i in range(2):
    for j in range(2):
         k+=1
         tx=axs[i,j].text(20,20,'',fontsize=15,va='top',
                          color='white')
         cont_label=contrast_array[k]
         tx.set_text('%g'%np.round(cont_label,1)+r'$\,\%$')
plt.show()
quit()

