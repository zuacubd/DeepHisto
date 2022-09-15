import os
import sys

def get_all_lines(dataset_path, header=False):
    """
        reads the list of lines in a file
    """
    try:
        with open(dataset_path, 'r') as fr:
            lines = fr.readlines()
    except Exception as e:
        print ("{} is in get_slides_path().".format(e))

    clean_lines = []
    if header:
        for idx in range(1, len(lines)):
            clean_lines.append(lines[idx].rstrip())
    else:
        clean_lines = [line.rstrip() for line in lines]

    return clean_lines


def get_slides_path(dataset_path):
    """ reads the list of path of the
        whole slides
    """
    try:
        with open(dataset_path, 'r') as fr:
            lines = fr.readlines()
    except Exception as e:
        print ("exception in get_slides_path() ... ")
        print (e)
    clines = [line.rstrip() for line in lines]
    return clines


def get_slide_class_path(dataset_path):
    """ reads the list of path of the whole slides plus
        their labels
    """
    try:
        with open(dataset_path, 'r') as fr:
            lines = fr.readlines()
    except Exception as e:
        print ("exception in get_slides_class_path() ... ")
        print (e)

    slidepath_class = {}
    for line in lines:
        cline = line.rstrip()
        parts = cline.split("\t")
        path = parts[0]
        cls = parts[1]
        slidepath_class[path] = cls
    return slidepath_class


def get_patch_class_path(dataset_path):
    """ reads the list of path of the patches plus
        their labels
    """
    try:
        with open(dataset_path, 'r') as fr:
            lines = fr.readlines()
    except Exception as e:
        print ("exception in get_patch_label_path() ... ")
        print (e)

    patchpath_class = {}
    for line in lines:
        cline = line.rstrip()
        parts = cline.split("\t")
        path = parts[0]
        cls = parts[1]
        patchpath_class[path] = cls

    return patchpath_class


def get_slide_patch_class_path(dataset_path):
    """ reads the list of path of the patches plus
        their labels
    """
    try:
        with open(dataset_path, 'r') as fr:
            lines = fr.readlines()
    except Exception as e:
        print ("exception in get_slide_patch_label_path() ... ")
        print (e)

    slidepath_patchpath_class = {}
    for line in lines:
        cline = line.rstrip()
        parts = cline.split("\t")
        slide_path = parts[0]
        patch_path = parts[1]
        cls = parts[2]
        patchpath_class = slidepath_patchpath_cls.get(slide_path)
        if patchpath_class is None:
            patchpath_class = {}
        patchpath_class[patch_path] = cls
        slidepath_patchpath_class[slide_path] = patchpath_cls

    return slidepath_patchpath_class
