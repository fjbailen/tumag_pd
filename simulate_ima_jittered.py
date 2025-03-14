"""
This function reads an MHD simulation and simulates the presence of jitter
of a given amplitude and infers it using minimization_jitter.
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
#plt.rcParams['figure.constrained_layout.use'] = True #For the layout to be as tight as possible
#plt.rcParams["image.interpolation"] = 'none'


#Parameters of input data
file='cadence' #MHD simulation to be opened. 'continuo' or 'cadence'
SNR=0 #Signal-to-noise ratio. 0 if no noise is to be applied
cobs=32.4 #Diameter of central obscuration as a percentage of the aperture
plate_scale=0.055 #Plate scale of the simulations in arcseconds (arcsec/pixel)
sigmax=0.1 #RMS of jitter along X direction (px)
sigmay=0.1 #RMS of jitter along X direction (px)
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
    N=250#256 #Number of pixels of the image
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
#sigma=np.array([rms_x,rms_y,0])
print('Inferred sigma (arcsec):\n',sigma)
#print('Sigma (px units):',sigma/plate_scale)
print('True sigma (arcsec):',rms_x,rms_y)

quit()

#Restore jittered image
ima_pad,pad_width=pdf.padding(ima_shift)
o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad,
                sigma,aberr,0,cobs=cobs,low_f=0.2,
                wind=True,reg1=0.05,reg2=1,inst='imax')
o_plot=o_plot[pad_width:-pad_width,pad_width:-pad_width]


#Restore jittered-free image
ima_pad_nojitter,pad_width=pdf.padding(ima)
o_plot_nojitter,_,noise_filt=pdf.object_estimate_jitter(ima_pad,#ima_pad_nojitter,
                0*sigma,aberr,0,cobs=cobs,low_f=0.2,
                wind=True,reg1=0.05,reg2=1,inst='imax')
o_plot_nojitter=o_plot_nojitter[pad_width:-pad_width,pad_width:-pad_width]


"""
Compute contrasts and plots
"""

#Contrasts
cima0=np.round(100*np.std(ima0)/np.mean(ima0),1) #Contrast of  MHD image
cima=np.round(100*np.std(ima)/np.mean(ima),1) #Contrast of jitter-free image
cima_shift=np.round(100*np.std(ima_shift)/np.mean(ima_shift),1) #Contrast of jittered image
c_rest=np.round(100*np.std(o_plot)/np.mean(o_plot),1) #Contrast of restored jittered image
c_rest_nojitter=np.round(100*np.std(o_plot_nojitter)/np.mean(o_plot_nojitter),1) #Contrast of restored jitter-free image


#MHD, jitter-free, jittered and restored images
fig,axs=plt.subplots(1,3)
axs[0].imshow(ima0,cmap='gray') #Jitter-free original MHD image
axs[1].imshow(ima,cmap='gray') #Jitter-free image after diffraction and noise
axs[2].imshow(ima_shift,cmap='gray') #Jittered image after diffraction and noise
axs[0].set_title('MHD simulation')
axs[1].set_title('Diffraction')
axs[2].set_title('Diffraction + jitter')
pf.remove_tick_labels(axs)
fig.set_size_inches(15,5)
plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99,hspace=0.0,
                    wspace=0.0)
txt_labels=['(a)','(b)','(c)']
cont_label=[cima0,cima,cima_shift]
for i in range(3):
    tx=axs[i].text(20,20,'',fontsize=20,va='top',
                          color='white')       
    tx.set_text(txt_labels[i])
    tx2=axs[i].text(N-60,20,'',fontsize=20,va='top',
                          color='white')       
    tx2.set_text('%g'%np.round(cont_label[i],1)+r'$\,\%$')

plt.show()
quit()

#Jitter-free, jittered and restored images
vmin=np.min(ima0)
vmax=np.max(ima0)
fig,axs=plt.subplots(2,2)
axs[0,0].imshow(ima0,cmap='gray',vmin=vmin,vmax=vmax) #Jitter-free image after diffraction and noise
axs[0,1].imshow(ima_shift,cmap='gray',vmin=vmin,vmax=vmax) #Jittered image after diffraction and noise
axs[1,1].imshow(o_plot_nojitter,cmap='gray',vmin=vmin,vmax=vmax) #Jitter-free restored image
axs[1,0].imshow(o_plot,cmap='gray',vmin=vmin,vmax=vmax) #Restored image
plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99,hspace=0.0,
                    wspace=0.0)
fig.set_size_inches(7,7)
pf.remove_tick_labels(axs)


#Add text labels with the contrast values for 2nd plot
contrast_array=np.array([cima0,cima_shift,c_rest,c_rest_nojitter])
k=-1
for i in range(2):
    for j in range(2):
         k+=1
         tx=axs[i,j].text(N-75,10,'',fontsize=15,va='top',
                          color='white')
         cont_label=contrast_array[k]
         tx.set_text('%g'%np.round(cont_label,1)+r'$\,\%$')


#MHD, jittered, jitter-free and restored images
vmin=None
vmax=None
fig,axs=plt.subplots(2,2)
axs[0,0].imshow(ima0,cmap='gray',vmin=vmin,vmax=vmax) #MHD
axs[0,1].imshow(ima,cmap='gray',vmin=vmin,vmax=vmax) #Jittered image after diffraction and noise
axs[1,1].imshow(ima_shift,cmap='gray',vmin=vmin,vmax=vmax) #Jitter-free restored image
axs[1,0].imshow(o_plot,cmap='gray',vmin=vmin,vmax=vmax) #Restored image
plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99,hspace=0.0,
                    wspace=0.0)
fig.set_size_inches(7,7)
pf.remove_tick_labels(axs)    

#Add text labels with the contrast values for 3rd plot
contrast_array=np.array([cima0,cima,c_rest,cima_shift])
label_subfigure=['(a)','(b)','(d)','(c)']
k=-1
for i in range(2):
    for j in range(2):
         k+=1
         tx=axs[i,j].text(N-75,10,'',fontsize=15,va='top',
                          color='white')
         tx2=axs[i,j].text(10,10,'',fontsize=15,va='top',
                          color='white')
         cont_label=contrast_array[k]
         tx.set_text('%g'%np.round(cont_label,1)+r'$\,\%$')
         tx2.set_text(label_subfigure[k])

plt.show()
quit()

