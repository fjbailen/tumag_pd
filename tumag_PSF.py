import sys
sys.path.append('./functions')
from matplotlib import pyplot as plt
import numpy as np
import pd_functions_v22 as pdf
import matplotlib.colors as colors

"""
Input parameters
"""
#Number of pixels of the PSF array
N=300 

#List of Zernike coefficients
#zernikes=np.zeros(4) #If we ignore aberrations
zernikes=np.array([ 0,0,0,0.41504611,0.20220434,-0.50903575,
  0.10019678,0.2121423,0.37931034,0.29463613,0.07773656,0.44473074,
 -0.41532383,0.31802193,-0.128198,-0.11317134,-0.33396265,0.07905502,
 -0.06274168,-0.1311398,0.08048971])

#Fraction of the telescope aperture obstructed by the central obscuration
cobs=32.4 

#Wavelength, f-number and pixel size of TuMag
wvl,fnum,Delta_x=pdf.tumag_params()

"""
Compute PSF
"""
#Critical frequency and pupil radius
nuc,R=pdf.compute_nuc(N,wvl,fnum,Delta_x)

#Polar coordinates
RHO,THETA=pdf.sampling2(N,R)

#Telescope aperture
ap=pdf.aperture(N,R,cobs=cobs)


#PSF
PSF=pdf.PSF(zernikes,0,RHO,THETA,ap)
PSF=PSF/np.max(PSF) #Normalization

#Radial PSF
xp0=int(PSF.shape[0]/2)
PSF_rad=pdf.radial_profile(PSF,[xp0,xp0])


#Plot in log scale
fig,ax=plt.subplots()
im=ax.imshow(PSF,norm=colors.LogNorm(vmin=1e-10))
fig.colorbar(im)

#Plot of radial PSF
fig,ax=plt.subplots()
ax.plot(PSF_rad)
ax.set_ylabel('PSF amplitude (a.u.)')
ax.set_xlabel('Radius (px)')
plt.show()
