from cv_utils.WSI import generate_training_patches

path_slide = 'WSI/slide/tumor/tumor_026.tif'
path_mask = 'WSI/mask/tumor/tumor_026.tif'
level = 0
patch_size = (300, 300)
stride = (128, 128)
inspection_size = (128, 128)
min_pct_tissue_area = 0.1
min_pct_tumor_area = 0.05
save_dir = {'tumor': 'WSI/patches/tumor',
            'normal': 'WSI/patches/normal'}
EXTENSION = 'tif'
OVERWRITE = False

generate_training_patches(path_slide, path_mask, level, patch_size, stride, inspection_size,
                          save_dir, drop_last=True, min_pct_tissue_area=min_pct_tissue_area,
                          min_pct_tumor_area=min_pct_tumor_area, ext=EXTENSION,
                          overwrite=OVERWRITE)
