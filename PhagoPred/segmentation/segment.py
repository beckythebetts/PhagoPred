from PhagoPred.detectron_segmentation import segment as detectron_segment
from PhagoPred.unet_segmentation import segment as unet_segment
from PhagoPred import SETTINGS


def main(dataset=SETTINGS.DATASET, maskrcnn=SETTINGS.MASK_RCNN_MODEL, unet=SETTINGS.UNET_MODEL):
    detectron_segment.main()
    # unet_segment.main()

if __name__ == '__main__':
    main()