import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import h5py

from PhagoPred import SETTINGS
from PhagoPred.utils import tools, mask_funcs
from PhagoPred.unet_segmentation.dataset import UNetDataset_inference, UNetDataset_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def segment_image(im_path, model=SETTINGS.UNET_MODEL):
    unet_dataset = UNetDataset_train()
def segment(dataset=SETTINGS.DATASET, model=SETTINGS.UNET_MODEL):
    with h5py.File(dataset, 'r+') as f:
        unet_dataset = UNetDataset_inference(f)
        unet_dataset_loader = torch.utils.data.DataLoader(unet_dataset, batch_size=1)

        model = torch.load(model / 'model.pth')
        model.eval()
        with torch.no_grad():
            for frame, image in enumerate(unet_dataset_loader):
                image = image.to(device)
                pred_mask = model(image)
                image = tools.unpad_image(image[0, 0, :, :], unet_dataset.padding_width, unet_dataset.padding_height).cpu().numpy()
                init_output = tools.unpad_image(pred_mask[0, 0, :, :], unet_dataset.padding_width, unet_dataset.padding_height).cpu().numpy()
                # print(pred_mask.shape)
                pred_mask = pred_mask > 0
                # print(pred_mask.shape)
                pred_mask = tools.unpad_image(pred_mask[0, 0, :, :], unet_dataset.padding_width, unet_dataset.padding_height).cpu().numpy()
                joined = np.concatenate((init_output, pred_mask, image), axis=1)
                plt.matshow(joined)
                plt.show()

                # if f'{int(frame):04}' in f['Segmentations']['Phase'].keys():
                #     current_mask = f['Segmentations']['Phase'][f'{int(frame):04}'][:]
                #     current_mask[(current_mask==0) & pred_mask] = 1
                #     # current_mask = pred_mask
                #     f['Segmentations']['Phase'][f'{int(frame):04}'][:] = current_mask
                # else:
                #     f['Segmentations']['Phase'].create_dataset(f'{int(frame):04}', data=pred_mask)

def main():
    segment()

if __name__ == '__main__':
    main()


