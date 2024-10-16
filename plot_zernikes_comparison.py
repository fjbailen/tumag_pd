import sys
sys.path.append('./functions')
from matplotlib import pyplot as plt
import numpy as np
import math_func2 as mf
import pd_functions_v17 as pdf
import os
import read_functions as rf
"""
Line command inputs: Jmax
Imports and plots the average of the Zernike coefficients
over all subpathes in the same figure for different PD optimizations
for comparison purposes
"""

Jmax=25#Maximum index of the zernike polynomials
Jmin=2 #Minimum index of the zernike polynomials
N0=1480 #Size of the focused/defocused images in the FITS file
magnif=2.47 #Magnification factor of TuMag



#Path and name of the FITS file containing the focused and defocused images
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='e2e_SIEMENS/' #Name of the folder containing th FITS file

fnamevec=['F4_W1_517_3_cam_1_focus_2840_defocus_3200',\
'P.D_W1_517_3_cam_1_focus_3200_defocus_2840']#['PHI5','PHI9','STP136']
colorvec=['r','b']
markervec=['o','^']
linevec=['solid','dashed']



fig4,axs4=plt.subplots()
ii=-1
for fname in fnamevec:
    ii+=1
    colori=colorvec[ii]
    markeri=markervec[ii]
    linei=linevec[ii]

    #Colormap limits for wavefront representation
    vmin=-np.pi #Typically, pi/2 or pi
    vmax=np.pi

    """
    Loop
    """
    av=np.zeros((Jmax-1,len(fnamevec)))
    rms_labels=[]


    """
    Results after optimization files
    """

    filename='a_optimized_Jmax_%g_*.txt'%(Jmax)
    folderpath=dir_folder+'txt/'+fname
    filetxt=rf.list_files(folderpath,filename)[0] #Assuming there is only one a_optimized file
    #Import txt files
    data=np.genfromtxt(folderpath+'/'+filetxt,delimiter='\t',unpack=False,\
    dtype=None,encoding='utf-8')
    names=np.array(data[:,0],dtype='str')
    values=data[:,1]

    #Obtain values from imported data
    a1_ind=np.argwhere(names=='a1')[0][0]
    a=np.array(values[a1_ind:],dtype='float64')
    av[:,ii]=a
    norm_a=2*np.pi/np.linalg.norm(a)

    maxit_ind=np.argwhere(names=='maxit')
    maxit=float(values[maxit_ind])
    wcut_ind=np.argwhere(names=='w_cut')
    w_cut=float(values[wcut_ind])



    norm_aver=2*np.pi/np.linalg.norm(av[:,ii])

    #Average zernike coefficients
    axs4.plot(range(Jmin,Jmax),av[(Jmin-1):,ii]/(2*np.pi),marker=markeri
    ,color=colori,label=fname,linestyle=linei,fillstyle='none')
    axs4.set_ylabel(r'Zernike coefs. [$\lambda$]')
    axs4.set_xlabel('Zernike index')

    print(fname,'rms:','1/%.3g waves'%norm_aver)
plt.legend(loc='lower right')
plt.show()
plt.close()

#Subtract Zernikes
if len(fnamevec)==2:
    a_difference=av[:,1]-av[:,0]
    norm_diff=2*np.pi/np.linalg.norm(a_difference)

    plt.plot(range(Jmin,Jmax),a_difference[(Jmin-1):]/(2*np.pi),marker='o'
    ,color='k',label='Aberrations introduced by PD plate')
    plt.ylabel(r'Zernike coefs. [$\lambda$]')
    plt.xlabel('Zernike index')
    print('PD plate rms wavefront error:','1/%.3g waves'%norm_diff)
    plt.show()
    plt.close()


"""
Save txt and images
"""

#Aberrations and parameters txt file
flabela=['ffolder','fname_F4','fname_PD']
for ai in range(len(a_difference)):
    flabela.append('a%g'%(ai+1))
param=np.array([ffolder,fnamevec[0],fnamevec[1]])

print(param.shape)
print(a_difference.shape)
filea=np.concatenate((param.reshape(len(param),1),a_difference.reshape(len(a_difference),1)))
filea=np.column_stack((flabela,filea))

filename='./txt/a_PD_Jmax_%g.txt'%(Jmax)
try:
    os.mkdir('./txt/')
except FileExistsError:
    print('./txt/'+' already created')
np.savetxt(filename,filea,delimiter='\t',fmt='%s',encoding='utf-8')
