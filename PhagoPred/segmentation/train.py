import shutil
from pathlib import Path

from PhagoPred.detectron_segmentation import train as detectron_train
from PhagoPred.unet_segmentation import train as unet_train
from PhagoPred import SETTINGS
from PhagoPred.utils import tools
"""
Start with Mask R-CNN type Dataset in segmentation:
{Model Name}
    train
        images
            {image_names}.png
        labels.json (COCO json format, with 'Cell' and 'Cluster' classes)
    validate
            "
"""
def main(maskrcnn=SETTINGS.MASK_RCNN_MODEL, unet=SETTINGS.UNET_MODEL, training_data=SETTINGS.TRAINING_DATA):
    for type in ('train', 'validate'):
    #     (maskrcnn / 'Training_Data' / type).mkdir(parents=True)
    #     tools.split_coco_by_category(training_data / type / 'labels.json', maskrcnn / 'Training_Data' / type , category='Macrophage')
    #     shutil.copytree(training_data / type / 'images', maskrcnn / 'Training_Data' / type / 'images')

        tools.coco_to_semantic_training_data(training_data  / type / 'images',
                                         training_data  / type / 'labels.json',
                                         unet / 'Training_Data' / type / 'images',
                                         unet / 'Training_Data' / type / 'masks')

    
    # detectron_train.train(maskrcnn)
    unet_train.train(unet)

if __name__ == '__main__':
    main()