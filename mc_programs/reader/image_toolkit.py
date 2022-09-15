import os
import sys

import numpy as np
import itertools
from openslide import OpenSlide

#from skimage import io, color

def get_wsi_header(wsi_path):
    """
        Read a whole slide and returns the slide object
    """
    try:
        slide_header = OpenSlide(wsi_path)
        #img = slide_header.read_region(location=(0,0), level=level, size=slide_header.level_dimensions[level])
        #img_rgb = np.asarray(img)[:,:,:-1]
        #return img_rgb
        return slide_header
    except Exception as e:
        print ("error in get_wsi(wsi_path, level)")
        print (e)

def get_wsi_slide(slide_header, level):
    """
        returns the patch in specific level
    """
    #rgba_image_pil = None
    try:
        rgba_image_pil = slide_header.read_region(location=(0,0), level=level, size=slide_header.level_dimensions[level])
        return rgba_image_pil
        #img_rgb = np.asarray(img)[:,:,:-1]
    except Exception as e:
        print ("error in get_wsi_patch ")
        print (e)
    #return rgba_image_pil


def get_wsi_patch(slide_header, x, y, level, width, height):
    """
        returns the patch in specific level
    """
    try:
        img_rgba_pil = slide_header.read_region(location=(x,y), level=level, size=(width, height))
        #img_rgb = np.asarray(img)[:,:,:-1]

    except Exception as e:
        print ("error in get_wsi_patch ")
        print (e)
    return img_rgba_pil
