import os
import sys

import math
import cv2
import numpy as np
import uuid
from PIL import Image

from reader.image_toolkit import get_wsi_header, get_wsi_slide, get_wsi_patch
from reader.dataset import get_slide_class_path, get_patch_class_path, get_slide_patch_class_path
from .wsi_roi import get_roi_bbox, get_patch_coords, get_rgba_to_rgb

def get_patch_extract(config, level, size, overlap, save_ext):
    """
        Extract patches for train and test slides images
    """

    #patches for training slides
    train_slide_path_file = config['train_slide_gc_ngc_path']
    train_patch_path_file = config['train_patch_gc_ngc_path']
    train_slide_patch_path_file = config['train_slide_patch_gc_ngc_path']
    train_patch_folder = config['train_patch_gc_ngc_folder']
    extract(config, train_slide_path_file, train_patch_path_file, train_slide_patch_path_file, train_patch_folder, \
            level, size, overlap, save_ext)

    #patches for testing slides
    test_slide_path_file = config['test_slide_gc_ngc_path']
    test_patch_path_file = config['test_patch_gc_ngc_path']
    test_slide_patch_path_file = config['test_slide_patch_gc_ngc_path']
    test_patch_folder = config['test_patch_gc_ngc_folder']

    extract(config, test_slide_path_file, test_patch_path_file, test_slide_patch_path_file, test_patch_folder, \
            level, size, overlap, save_ext)


def extract(config, slides_path_file, patch_path_file, slide_patch_path_file, patch_folder, \
            level, size, overlap, save_ext):
    """
        Extract patches from slides
    """

    slides_class = get_slide_class_path(slides_path_file)
    count = 0
    level_bbox = 3
    level_use = 0
    mag_factor_bbox = 1<<level_bbox
    mag_factor_use = 1<<level_use
    mag_factor = int((mag_factor_bbox*1.0)/mag_factor_use)
    patch_label_list = []
    slide_patch_label = {}
    total_pix = 3*size*size

    patch_folder = os.path.join(patch_folder, 'level'+str(level_use))
    if not os.path.exists(patch_folder):
        os.makedirs(patch_folder)

    grid4 = ['00', '01', '10', '11']

    for slide_path in slides_class:
        slide_cls = slides_class.get(slide_path)
        slide_label = config.get(slide_cls)

        print ("{0}:{1}:{2}".format(slide_path, slide_cls, slide_label))
        slide_header = get_wsi_header(slide_path)
        dimensions = slide_header.level_dimensions[level]
        width, height = dimensions[0], dimensions[1]
        print ("(Width, Height) = ({0}, {1})".format(width, height))

        print ("Get a slide at level {0}".format(level_bbox))
        rgba_pil = get_wsi_slide(slide_header, level_bbox)
        print ("(Width, Height) = {0}".format(rgba_pil.size))

        print ("Get the best five region of interest (roi) at level {0}".format(level_bbox))
        bboxes = get_roi_bbox(rgba_pil)
        filename = os.path.splitext(os.path.basename(slide_path))[0]

        extract_patch(slide_header, level_use, bboxes, mag_factor, grid4, slide_label, slide_path, \
                      slide_patch_label, patch_label_list):

        count = count + 1
        if count == 1:
            break

    with open(patch_path_file,'w') as fw:
        for x in patch_label_list:
            fw.write(x+'\n')
        fw.close()

    with open(slide_patch_path_file, 'w') as fw:
        for slide in slide_patch_label:
            patch_label = slide_patch_label.get(slide)
            for x in patch_label:
                fw.write(slide + '\t' + x + '\n')
        fw.close()


def extract_patch(slide_header, level_use, bboxes, mag_factor, grids, slide_label, slide_path, \
                  slide_patch_label, patch_label_list):
    """ Extract patches from high resolution slides
        given a bounding box in low resolution
    """

    for bbox in bboxes:
        x, y, w, h = bbox
        print ("Bbox size: (({0},{1}), {2}, {3})".format(x, y, w, h))
        box_name = str(x)+"_"+str(y)+"_"+str(w)+"_"+str(h)

        x_begin = x * mag_factor
        y_begin = y * mag_factor
        width = w * mag_factor
        height = h * mag_factor
        half_width = math.floor(width/2)
        half_height = math.floor(height/2)

        for grid in grids:
            x_delta = int(grid[0]) * half_width
            y_delta = int(grid[1]) * half_height

            x_hr = x_begin + x_delta
            y_hr = y_begin + y_delta

            #read_region((x,y), level, (width, height))
            print ("Get a slide at level ({0} of ({1},{2}), {3}, {4})".format(level_use, x_hr, y_hr, half_width, half_height))
            rgba_pil = get_wsi_patch(slide_header, x_hr, y_hr, level_use, half_width, half_height)
            print("Shape is: {0}".format(rgba_pil.size))

            rgba_image = np.asarray(rgba_pil)
            print("Shape of rgba_image: {0}".format(rgba_image.shape))

            rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)
            print("Shape of rgb_image: {0}".format(rgb_image.shape))

            #print ("Get patch coordinates at level {0}".format(level))
            #patch_coors = get_patch_coords(bboxes, mag_factor, size, overlap)
            b_x_start = 0
            b_x_end = rgb_image.shape[1]
            b_y_start = 0
            b_y_end = rgb_image.shape[0]
            X = np.arange(b_x_start, b_x_end, step=size)
            Y = np.arange(b_y_start, b_y_end, step=size)
            print("Length of X: {0}".format(len(X)))
            print("Length of Y: {0}".format(len(Y)))

            for x_width in X:
                for y_height in Y:

                    bbox_rgb_patch = rgb_image[y_height:(y_height+size), x_width:(x_width+size), :]
                    print("rgb_path:({0},{1})={2}".format(size, size, bbox_rgb_patch.shape))
                    total_pix = 3*bbox_rgb_patch.shape[0]*bbox_rgb_patch.shape[1]
                    no_black_pix = np.sum(bbox_rgb_patch == 0)
                    percentage = int((no_black_pix*100)/total_pix)
                    if percentage >= 25:
                        continue

                    bbox_rgb_patch_pil = Image.fromarray(bbox_rgb_patch)
                    patch_name = "{}_{}_{}_{}x{}_{}.{}".format(filename, box_name, grid, str(x_width), str(y_height), \
                                                            str(uuid.uuid4())[:8], save_ext)
                    print ("Patch name:{0}".format(patch_name))
                    patch_path = os.path.join(patch_folder, patch_name)
                    bbox_rgb_patch_pil.save(patch_path)

                    patch_label_list.append(patch_path + '\t' + str(slide_label))
                    patch_label = slide_patch_label.get(slide_path)
                    if patch_label is None:
                        patch_label = []
                    patch_label.append(patch_path + '\t' + str(slide_label))
                    slide_patch_label[slide_path] = patch_label

         print ("{0} Grid processed".format(grid))

    print ("Patch extract is done for (({0},{1}), {2}, {3}) bbox".format(x, y, w, h))
