#A script keeping track of the path

config_class_level = {
    'GC':2,
    'NGC':1,
    'BG':0,
    'train_slides_gc_ngc_path':'dataset/train_slides_class_level_annotated_gc_ngc_path.txt',
    'train_patch_gc_ngc_bg_path':'dataset/train_patch_gc_ngc_bg_path.txt',
    'train_slides_patch_gc_ngc_bg_path':'dataset/train_slides_patch_gc_ngc_bg_path.txt',
    'train_patch_gc_ngc_bg_folder':'/projets/sig/mullah/oncopole/slides/class_level/mc/train',
    'train_patch_hr_gc_ngc_bg_path':'dataset/train_patch_hr_gc_ngc_bg_path.txt',
    'train_slides_patch_hr_gc_ngc_bg_path':'dataset/train_slides_patch_hr_gc_ngc_bg_path.txt',
    'train_patch_hr_gc_ngc_bg_folder':'/projets/sig/mullah/oncopole/slides/class_level/hr/train',
    'test_slides_gc_ngc_path':'dataset/test_slides_class_level_annotated_gc_ngc_path.txt',
    'test_patch_gc_ngc_bg_path':'dataset/test_patch_gc_ngc_bg_path.txt',
    'test_slides_patch_gc_ngc_bg_path':'dataset/test_slides_patch_gc_ngc_bg_path.txt',
    'test_patch_gc_ngc_bg_folder':'/projets/sig/mullah/oncopole/slides/class_level/mc/test',
    'test_patch_hr_gc_ngc_bg_path':'dataset/test_patch_hr_gc_ngc_bg_path.txt',
    'test_slides_patch_hr_gc_ngc_bg_path':'dataset/test_slides_patch_hr_gc_ngc_bg_path.txt',
    'test_patch_hr_gc_ngc_bg_folder':'/projets/sig/mullah/oncopole/slides/class_level/hr/test'
}

config_patch_level = {
    'train_slides_gc_ngc_path':'dataset/train_slides_patch_level_annotated_gc_ngc_path.txt',
    'train_patch_gc_ngc_path':'dataset/train_patch_gc_ngc.txt',
    'train_patch_gc_ngc_folder':'/projets/sig/mullah/oncopole/slides/class_level/train',
    'test_slides_gc_ngc_path':'dataset/test_slides_patch_level_annotated_gc_ngc_path.txt',
    'test_patch_gc_ngc_path':'dataset/test_patch_gc_ngc.txt',
    'test_patch_gc_ngc_folder':'/projets/sig/mullah/oncopole/slides/class_level/test'
}


train_parameters = {
    #'learning_rate':0.001,
    'learning_rate':0.0001,
    'num_epochs':20,
    'batch_size':32,
    'dropout_rate':0.5,
    'num_classes':3,
    'display_step':20,
    'save_step':2000
}

output_parameters = {
    'prediction_folder':'output/prediction',
    'result_folder':'output/results'
}

imagenet_pretrain_weights = {
    'alexnet':'imagenet_models/bvlc_alexnet.npy'
}
