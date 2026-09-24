import os
import re
from PIL import Image, ImageDraw, ImageFont

import numpy as np

def make_montages(input_folder, output_folder, minutes_per_frame=10):
    os.makedirs(output_folder, exist_ok=True)

    # Regex to extract fields from filename
    pattern = re.compile(r"(\d+)_(\d+)_(\d+)_(alive|dead)\.jpe?g", re.IGNORECASE)

    # Group by cell_idx
    cell_images = {}
    for fname in os.listdir(input_folder):
        match = pattern.match(fname)
        if match:
            cell_idx, frame, other, status = match.groups()
            cell_idx, frame = int(cell_idx), int(frame)
            cell_images.setdefault(cell_idx, []).append((frame, status, fname))

    # Try to use Arial; fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for cell_idx, images in sorted(cell_images.items()):
        # Sort by frame number
        images.sort(key=lambda x: x[0])
        frames = [int(image[0]) for image in images]
        min_frame = min(frames)
        
        # Load all images
        loaded_imgs = [Image.open(os.path.join(input_folder, f)).convert("RGB") for _, _, f in images]
        max_h = max(img.height for img in loaded_imgs)
        margin = 40  # space for labels

        montage_w = sum(img.width for img in loaded_imgs)  # sum of actual widths
        montage_h = max_h + margin
        montage = Image.new("RGB", (montage_w, montage_h), (255, 255, 255))
        draw = ImageDraw.Draw(montage)

        x_cursor = 0
        for i, (frame, status, fname) in enumerate(images):
            img = loaded_imgs[i]
            y_offset = (max_h - img.height) // 2  # vertical centering
            montage.paste(img, (x_cursor, y_offset))

            # label
            minutes = (frame - min_frame) * minutes_per_frame
            label = f"{minutes} min ({status})"
            bbox = draw.textbbox((0,0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = x_cursor + (img.width - text_w) // 2
            text_y = max_h + 5
            draw.text((text_x, text_y), label, fill=(0,0,0), font=font)

            x_cursor += img.width  # move cursor for next image

            # --- Add scale bar ---
        scale_length_um = 20  # scale bar length in microns
        px_per_um = 1 / 0.325   # 1 px = 6.5 µm
        scale_length_px = int(scale_length_um * px_per_um)

        bar_height = 5  # pixels
        margin = 20     # margin from bottom-left corner

        # Top-right coordinates
        bar_x1 = montage_w - margin - scale_length_px
        bar_y1 = margin
        bar_x2 = montage_w - margin
        bar_y2 = margin + bar_height

        # Draw the scale bar rectangle
        draw.rectangle([bar_x1, bar_y1, bar_x2, bar_y2], fill=(0,0,0))

        # Draw the label above the bar
        label = f"{scale_length_um} um"
        bbox = draw.textbbox((0,0), label, font=font)
        text_x = montage_w - margin - scale_length_px  # align with left of bar
        text_y = bar_y1 - (bbox[3]-bbox[1]) - 10        # slightly above the bar
        draw.text((text_x, text_y), label, fill=(0,0,0), font=font)
        
        # Save montage
        out_path = os.path.join(output_folder, f"cell_{cell_idx}_montage.jpeg")
        montage.save(out_path)
        print(f"Saved montage for cell {cell_idx} → {out_path}")
        
if __name__ == "__main__":
    input_folder = "PhagoPred/detectron_segmentation/models/27_05_mac/Fine_Tune_Data_all_03_10/images"
    output_folder = "temp/montages"
    make_montages(input_folder, output_folder, minutes_per_frame=1)