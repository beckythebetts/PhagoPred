import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import h5py
import numpy as np
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from PhagoPred.utils import mask_funcs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gui:
    def __init__(self, hdf5dataset):
        self.get_data(hdf5dataset)
        self.get_merged_images()
        self.get_tracked_images()
        #self.get_tracked_images()
        self.current_image_index = 0
        self.show_tracked_images = False

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Timelapse")
        self.root.geometry('500x600')
        self.root.resizable(width=True, height=True)

        # Create a canvas to display images
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Create a vertical scrollbar linked to the canvas
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=self.scrollbar.set)

        # Create a slider
        self.slider = tk.Scale(self.root, from_=0, to=len(self.merged) - 1, orient="horizontal", command=self.update_image)
        self.slider.pack(side=tk.BOTTOM, fill=tk.X)

        self.toggle_button = tk.Button(self.root, text='Show Tracking', command=self.toggle_image_display)
        self.toggle_button.pack(side=tk.BOTTOM, pady=10) 
        # Bind the resize event
        self.root.bind("<Configure>", self.on_resize)

        # Initialize with the first image
        self.update_image(self.slider.get())

        # Run the GUI
        self.root.mainloop()

    def update_image(self, val):
        self.current_image_index = int(val)
        img_data = self.tracked if self.show_tracked_images else self.merged
        img = Image.fromarray(img_data[self.current_image_index], 'RGB')

        # Resize the image to fit the window size
        img = self.resize_image_to_fit(img)

        # Convert the image to a format Tkinter can display
        self.tk_image = ImageTk.PhotoImage(img)

        # Update the canvas with the new image
        self.canvas.delete("all")  # Clear the canvas before drawing new image
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        self.canvas.image = self.tk_image

        # Update the canvas scroll region
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def toggle_image_display(self):
        self.show_tracked_images = not self.show_tracked_images
        new_text = "Show Raw Images" if self.show_tracked_images else "Show Tracked Images"
        self.toggle_button.config(text=new_text)
        self.update_image(self.slider.get())

    def on_resize(self, event):
        # Update image when window is resized
        self.update_image(self.slider.get())

    def resize_image_to_fit(self, image):
        max_width = max(self.root.winfo_width(), 1)  # Ensure non-zero dimensions
        max_height = max(self.root.winfo_height(), 1)  # Ensure non-zero dimensions
        img_width, img_height = image.size
        ratio = min(max_width / img_width, max_height / img_height)

        # Calculate new dimensions ensuring they're greater than zero
        new_width = max(int(img_width * ratio), 1)
        new_height = max(int(img_height * ratio), 1)

        return image.resize((new_width, new_height))

    def get_data(self, dataset):
        with h5py.File(dataset, 'r') as f:
            self.phase_data = np.array([f['Images']['Phase'][frame][:] for frame in list(f['Images']['Phase'].keys())], dtype='uint8')
            self.epi_data = np.array([f['Segmentations']['Epi'][frame][:] for frame in list(f['Segmentations']['Epi'].keys())], dtype='uint8')
            self.segmentation_data = np.array([f['Segmentations']['Phase'][frame][:] for frame in list(f['Segmentations']['Phase'].keys())], dtype='int16')
            self.max_cell_index = 0
            for frame in f['Segmentations']['Phase']:
                frame = np.array(f['Segmentations']['Phase'][frame])
                max = np.max(frame)
                if max > self.max_cell_index:
                    self.max_cell_index = max

    def get_merged_images(self):
        print('\nGETTING MERGED IMAGES\n')
        # self.epi_data[self.epi_data > 0] = 255
        # epi_channel = self.make_rgb(self.epi_data)
        # epi_channel[:, :, :, 1:3] = 0
        self.merged = self.make_rgb(self.phase_data)
        self.merged[:, :, :, 1][self.epi_data > 0] = 0
        self.merged[:, :, :, 2][self.epi_data > 0] = 0
        # self.merged = ((self.make_rgb(self.phase_data).astype(np.float32) + epi_channel.astype(np.float32)) / 2).astype(np.uint8)
        # print(self.merged.shape)

    def make_rgb(self, data):
        # Assuming this is a helper function to convert grayscale data to RGB
        return np.stack([data] * 3, axis=-1)

    def get_tracked_images(self):
        print('Getting tracked images')
        max_cell_index = np.max(self.segmentation_data)
        LUT = torch.randint(low=10, high=255, size=(max_cell_index+1, 3)).to(device)
        LUT[0] = torch.tensor([0, 0, 0]).to(device)
        rgb_phase = self.make_rgb(self.phase_data)
        # rgb_phase = np.stack((self.phase_data, phase_data, phase_data), axis=-1)
        self.tracked = np.zeros(rgb_phase.shape, dtype=np.uint8)
        for i, (phase_image, segmentation) in enumerate(
                zip(rgb_phase, self.segmentation_data)):
            sys.stdout.write(f'\rReading frame {i + 1}')
            sys.stdout.flush()
            segmentation = torch.tensor(segmentation).to(device)
            phase_image = torch.tensor(phase_image).to(device).int()
            sys.stdout.write(
                f'\rFrame {i + 1}')
            sys.stdout.flush()
            outlines = mask_funcs.mask_outlines(segmentation, thickness=3).int()
            outlines = LUT[outlines].type(torch.int16)
            
            phase_image = torch.where(outlines>0, outlines, phase_image)
            # plt.matshow(phase_image.clone().detach().cpu().numpy())
            # plt.show()
            phase_image = phase_image.cpu().numpy().astype(np.uint8)
            # plt.matshow(phase_image)
            # plt.show()
            self.tracked[i] = phase_image
        print(self.tracked.shape)

if __name__ == "__main__":
    gui = Gui(Path('PhagoPred') / 'Datasets' / 'no_filter01_short_tracked (copy).h5')
    gui.create_gui()

