import cv2
import numpy as np
import itertools

def get_roi_bbox(rgba_image_pil, num_roi):
    '''
        find the larget bbox in the wsi to ease the extraction of
        patches
    '''
    wsi_rgba = np.asarray(rgba_image_pil) #Pil to numpy array
    print ("RGBA_image size:{0}".format(wsi_rgba.shape))
    wsi_rgb = cv2.cvtColor(wsi_rgba, cv2.COLOR_RGBA2RGB) #RGBA to RGA (excluding the last channel)
    wsi_bgr = cv2.cvtColor(wsi_rgb, cv2.COLOR_RGB2BGR) #RGB to BGR for easing the cv2 operation
    wsi_hsv = cv2.cvtColor(wsi_bgr, cv2.COLOR_BGR2HSV) #BGR to HSV

    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    thresh = cv2.inRange(wsi_hsv, lower_red, upper_red) #HSV image threshold

    # print("Closing step: ")
    close_kernel = np.ones((15, 15), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(thresh), cv2.MORPH_CLOSE, close_kernel)

    # print("Openning step: ")
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(np.array(image_open), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if num_roi > 0:
        #Best contors by area
        contours_ = sorted(contours, key = cv2.contourArea, reverse = True)
        contours_ = contours_[:num_roi] #Top 5 contours
    else:
        contours_ = contours

    bboxes = [cv2.boundingRect(c) for c in contours_]
    return bboxes


def get_patch_coords(bboxes, mag_factor, size, overlap=False):
    '''
        design patch coordinates
        using the bboxes, magnification factor (* 2^level), and size corresponding to the level 0 slide
    '''
    coors_all = []
    def get_coors_overlap(begin, wsi_len, size):
        parts = []
        if wsi_len<size:
            return parts

        if wsi_len == size:
            parts.append(begin)
            return parts

        num_parts = int(np.ceil((wsi_len*1.0)/size))
        overlap_len = (num_parts*size - wsi_len)/(num_parts - 1)
        extend_len = (size - overlap_len)
        parts = [(begin + int(round(idx*extend_len))) for idx in np.arange(num_parts)]
        return parts

    def get_coors_no_overlap(begin, wsi_len, size):
        parts = []
        if wsi_len<size:
            return parts

        if wsi_len == size:
            parts.append(begin)
            return parts

        num_parts = int(np.floor((wsi_len*1.0)/size))
        parts = [(begin + idx*size) for idx in np.arange(num_parts)]
        return parts

    for i, box in enumerate(bboxes):
        x, y, w, h = box

        #x_begin = int(x) * mag_factor
        #y_begin = int(y) * mag_factor
        x_begin = 0
        y_begin = 0
        wsi_width = int(w) * mag_factor
        wsi_height = int(h) * mag_factor

        if overlap:
            w_points = get_coors_overlap(x_begin, wsi_width, size)
            h_points = get_coors_overlap(y_begin, wsi_height, size)
        else:
            w_points = get_coors_no_overlap(x_begin, wsi_width, size)
            h_points = get_coors_no_overlap(y_begin, wsi_height, size)

        if len(h_points)>0 and len(w_points)>0:
            coors = list(itertools.product(h_points, w_points))
            coors_all.extend(coors)
        #else:
        #    raise Exception("Error in get_patch_coors")

    return coors_all

def get_rgba_to_rgb(rgba_image):
    '''
        extracting rgb image from rgba image
    '''
    r_, g_, b_, a_ = cv2.split(rgba_image)
    rgb_image = cv2.merge((r_, g_, b_))
    return rgb_image
