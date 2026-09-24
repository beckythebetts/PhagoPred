# from backbones_unet.model.unet import Unet
# from backbones_unet.utils.dataset import SemanticSegmentationDataset
# from backbones_unet.model.losses import DiceLoss
# from backbones_unet.utils.trainer import Trainer
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from PhagoPred import SETTINGS
from PhagoPred.utils import tools
from PhagoPred.unet_segmentation.dataset import UNetDataset_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
def train(dir=SETTINGS.UNET_MODEL):
    train_ims_path = dir /  'Training_Data' / 'train' / 'images'
    train_masks_path = dir /  'Training_Data' / 'train' / 'masks'

    val_ims_path = dir /  'Training_Data' / 'validate' / 'images'
    val_masks_path = dir /  'Training_Data' / 'validate' / 'masks'

    train_dataset = UNetDataset_train(train_ims_path, train_masks_path)
    val_dataset = UNetDataset_train(val_ims_path, val_masks_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2)

    model = smp.Unet(
        backbone='resnet50',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1
    )

    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4) 

    epochs = 10  # Number of epochs to train

    with open(dir / 'training_losses.txt', 'w') as file:
        file.write('Epoch\tTraining Loss\tValidation Loss\n')

        for epoch in range(epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0

            for images, masks in train_loader:
                print(images.size())
                # Move tensors to GPU if available
                images, masks = images.to(device), masks.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)

                # Compute loss (you might want to apply sigmoid before calculating BCE if needed)
                loss = criterion(outputs, masks)

                # Backward pass
                loss.backward()

                # Optimize the model
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss  = running_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    loss = criterion(model(images), masks)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            file.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\n")


            print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    losses = pd.read_csv(dir / 'training_losses.txt', sep='\t', header=0)
    plt.clf()
    plt.rcParams["font.family"] = 'serif'
    plt.scatter(losses['Epoch'], losses['Training Loss'], color='navy')
    plt.scatter(
        losses['Epoch'], losses['Validation Loss'], color='red')
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.savefig(dir / 'loss_plot.png')
    plt.clf()
    torch.save(model, dir / 'model.pth')


# def train(dir=SETTINGS.UNET_MODEL):
#     train_ims_path = dir /  'Training_Data' / 'train' / 'images'
#     train_masks_path = dir /  'Training_Data' / 'train' / 'masks'

#     val_ims_path = dir /  'Training_Data' / 'validate' / 'images'
#     val_masks_path = dir /  'Training_Data' / 'validate' / 'masks'

#     train_dataset = SemanticSegmentationDataset(train_ims_path, train_masks_path, size=(896,1344))
#     val_dataset = SemanticSegmentationDataset(val_ims_path, val_masks_path, size=(896,1344))

#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
#     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2)

#     model = Unet(
#         backbone='resnet50',
#         in_channels=3,
#         num_classes=1
#     )

#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.AdamW(params, 1e-4) 

#     trainer = Trainer(
#         model,                    # UNet model with pretrained backbone
#         criterion=DiceLoss(),     # loss function for model convergence
#         optimizer=optimizer,      # optimizer for regularization
#         epochs=10                 # number of epochs for model training
#     )

#     trainer.fit(train_loader, val_loader)

def main():
    train()

if __name__ == '__main__':
    main()