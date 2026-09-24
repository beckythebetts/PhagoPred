import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from PhagoPred import SETTINGS
from PhagoPred.utils import tools, mask_funcs
from PhagoPred.unet_segmentation.dataset import UNetDataset_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def eval(dir=SETTINGS.UNET_MODEL):
    print(f'Evaluating {str(dir)} ...')
    val_ims_path = dir /  'Training_Data' / 'validate' / 'images'
    val_masks_path = dir /  'Training_Data' / 'validate' / 'masks'

    val_dataset = UNetDataset_train(val_ims_path, val_masks_path)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    model = torch.load(dir / 'model.pth')
    
    model.eval()

    with torch.no_grad():
        ious = [] 
        for i, (image, mask) in enumerate(val_loader):
            image, mask = image.to(device), mask.to(device)
            pred = model(image)
            binary_pred = pred.clone()
            binary_pred[binary_pred > 0] = 1
            binary_pred[binary_pred < 0] = 0

            mask, binary_pred = mask.type(torch.bool), binary_pred.type(torch.bool)
            view = tools.show_semantic_segmentation(image[0, 0, :, :], binary_pred[0, 0, :, :], mask[0, 0, :, :]).cpu().numpy().astype(np.uint8)
            view = Image.fromarray(view, 'RGB')
            view.save(dir / f'view_segmentation_{i}.png')

            view_output = tools.min_max_normalise(pred[0,0,:,:].cpu().numpy())
            view_output = Image.fromarray((view_output*255).astype(np.uint8), 'L')
            view_output.save(dir / f'view_output_{i}.png')
            

            ious.append(mask_funcs.cal_iou(binary_pred.cpu().numpy()[0], mask.cpu().numpy()[0]))
        with open(dir / 'iou.txt', 'w') as f:
            f.write('\t'.join(map(str, ious)) + f'\n{np.mean(ious):.4f} +- {np.std(ious):.4f}')
    print(f'\nIOU: {np.mean(ious)} +- {np.std(ious)}')
            
