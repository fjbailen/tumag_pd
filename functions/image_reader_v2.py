# ============================ IMPORTS ====================================== #

import os
import numpy as np
import glob
import matplotlib.pyplot as plt

from alive_progress import alive_bar
import datetime as dt

# Own Libs
import utils_mps as ut

# ============================= CONFIG ====================================== #

#plt.style.use('dark_background')

# =========================================================================== #

def Read_Image(path, PlotFlag = False, printHeader_flag = False, vmin = 0, vmax = 4000):

    # Separate Header and image data
    H, hl, I = ut.HeadernImageSeparator(path) # Header, header_lenth and image

    # Read header
    Head = ut.GetDatafromHeader(H)

    # Read Image
    Im  = ut.read_image(path, Head['Roi_x_size'], Head['Roi_y_size'], hl)

    if PlotFlag:
        plt.figure()
        im = plt.imshow(Im.T, cmap = 'inferno', vmin = vmin, vmax = vmax)
        plt.colorbar(im)

    if printHeader_flag:
        for key in Head:
            print(key, ' : ', Head[key])

    return Head, Im

def Read_Header(path, printHeader_flag = False):

    # Separate Header and image data
    H, hl, I = ut.HeadernImageSeparator(path) # Header, header_lenth and image

    # Read header
    Head = ut.GetDatafromHeader(H)

    # Read Image

    if printHeader_flag:
        for key in Head:
            print(key, ' : ', Head[key])

    return Head

def Folder_Data_reader(path):

    Images_paths = glob.glob(os.path.join(path, '*.img'))

    DATA = {}

    for img in Images_paths:
        image_name = os.path.basename(img)
        H, I = Read_Image(image_name)
        DATA[image_name] = H

    return DATA
