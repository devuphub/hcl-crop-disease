"""
Dataset Builder and Image Generator for Crop Disease Detection
Creates structured datasets across 18 crop disease classes with realistic visual characteristics,
and populates curated test sample images for immediate demonstration.
"""

import os
import random
import math
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice___Brown_Spot",
    "Rice___Leaf_Blast",
    "Rice___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]


def _create_base_leaf(crop_type: str, width: int = 224, height: int = 224) -> Image.Image:
    """
    Renders realistic base leaf geometry, shading, and venation for a given crop.
    """
    # Background: agricultural field / lab neutral background
    bg_color = (
        random.randint(220, 240),
        random.randint(215, 235),
        random.randint(200, 220)
    )
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    if crop_type in ["Apple", "Potato", "Tomato"]:
        # Broad ovate/palmate leaf
        leaf_color = (
            random.randint(35, 60),
            random.randint(110, 165),
            random.randint(30, 55)
        )
        rx = random.randint(70, 85)
        ry = random.randint(85, 100)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=leaf_color)

        # Central main vein and lateral secondary veins
        vein_color = (
            int(leaf_color[0] * 1.2),
            int(min(255, leaf_color[1] * 1.25)),
            int(leaf_color[2] * 1.2)
        )
        draw.line([cx, cy - ry + 10, cx, cy + ry - 10], fill=vein_color, width=3)
        for offset in range(-ry + 30, ry - 30, 22):
            draw.line([cx, cy + offset, cx - int(rx * 0.7), cy + offset - 15], fill=vein_color, width=2)
            draw.line([cx, cy + offset, cx + int(rx * 0.7), cy + offset - 15], fill=vein_color, width=2)

    elif crop_type in ["Corn", "Rice"]:
        # Slender linear / lanceolate blade
        leaf_color = (
            random.randint(45, 75),
            random.randint(125, 175),
            random.randint(25, 50)
        )
        rx = random.randint(35, 55)
        ry = random.randint(95, 108)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=leaf_color)

        # Parallel venation
        vein_color = (
            int(leaf_color[0] * 1.15),
            int(min(255, leaf_color[1] * 1.2)),
            int(leaf_color[2] * 1.15)
        )
        for vx in range(cx - rx + 10, cx + rx - 10, 8):
            draw.line([vx, cy - ry + 15, vx, cy + ry - 15], fill=vein_color, width=1)

    return img


def _apply_disease_symptoms(img: Image.Image, class_name: str) -> Image.Image:
    """
    Applies characteristic visual pathology patterns (lesions, rings, spots, chlorosis).
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy = w // 2, h // 2

    crop, disease = class_name.split("___")

    if disease == "healthy":
        # Add subtle natural texture variations
        return img.filter(ImageFilter.SMOOTH)

    elif "Early_blight" in disease:
        # Concentric dark brown target rings with yellow halo
        num_spots = random.randint(4, 8)
        for _ in range(num_spots):
            sx = cx + random.randint(-50, 50)
            sy = cy + random.randint(-55, 55)
            r = random.randint(14, 26)

            # Yellow chlorotic halo
            halo_color = (190, 185, 40)
            draw.ellipse([sx - r - 4, sy - r - 4, sx + r + 4, sy + r + 4], fill=halo_color)

            # Concentric rings
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(70, 45, 25))
            draw.ellipse([sx - int(r*0.7), sy - int(r*0.7), sx + int(r*0.7), sy + int(r*0.7)], fill=(110, 70, 35))
            draw.ellipse([sx - int(r*0.4), sy - int(r*0.4), sx + int(r*0.4), sy + int(r*0.4)], fill=(50, 30, 15))

    elif "Late_blight" in disease:
        # Irregular water-soaked dark greenish-black necrotic lesions
        num_patches = random.randint(3, 6)
        for _ in range(num_patches):
            px = cx + random.randint(-50, 50)
            py = cy + random.randint(-50, 50)
            rad = random.randint(22, 40)
            lesion_color = (40, 45, 30)
            draw.ellipse([px - rad, py - rad + 5, px + rad, py + rad - 5], fill=lesion_color)
            # White fuzzy edge sporulation
            draw.arc([px - rad, py - rad + 5, px + rad, py + rad - 5], start=0, end=180, fill=(220, 225, 215), width=2)

    elif "Apple_scab" in disease:
        # Olive-black velvety irregular corky spots
        num_scabs = random.randint(6, 12)
        for _ in range(num_scabs):
            sx = cx + random.randint(-55, 55)
            sy = cy + random.randint(-60, 60)
            sr = random.randint(8, 16)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(50, 60, 40))

    elif "Black_rot" in disease:
        # Frogeye leaf spot: purple-red margin with tan-brown center
        num_spots = random.randint(5, 10)
        for _ in range(num_spots):
            bx = cx + random.randint(-50, 50)
            by = cy + random.randint(-55, 55)
            br = random.randint(10, 20)
            draw.ellipse([bx - br - 2, by - br - 2, bx + br + 2, by + br + 2], fill=(110, 30, 50))
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(140, 100, 60))
            draw.ellipse([bx - 3, by - 3, bx + 3, by + 3], fill=(40, 20, 15))

    elif "Cedar_apple_rust" in disease or "Common_rust" in disease:
        # Bright rust-orange powdery spots / pustules
        num_pustules = random.randint(15, 35)
        for _ in range(num_pustules):
            rx = cx + random.randint(-55, 55)
            ry = cy + random.randint(-70, 70)
            rr = random.randint(3, 8)
            draw.ellipse([rx - rr, ry - rr, rx + rr, ry + rr], fill=(215, 100, 20))
            draw.ellipse([rx - 1, ry - 1, rx + 1, ry + 1], fill=(245, 175, 45))

    elif "Northern_Leaf_Blight" in disease:
        # Long cigar-shaped tan lesions along blade
        for _ in range(random.randint(2, 5)):
            lx = cx + random.randint(-25, 25)
            ly = cy + random.randint(-40, 40)
            draw.ellipse([lx - 12, ly - 35, lx + 12, ly + 35], fill=(160, 135, 95))
            draw.ellipse([lx - 7, ly - 25, lx + 7, ly + 25], fill=(120, 95, 65))

    elif "Brown_Spot" in disease:
        # Oval sesame-seed shaped brown spots with yellow margins
        for _ in range(random.randint(12, 24)):
            sx = cx + random.randint(-30, 30)
            sy = cy + random.randint(-70, 70)
            draw.ellipse([sx - 4, sy - 8, sx + 4, sy + 8], fill=(185, 170, 45))
            draw.ellipse([sx - 2, sy - 5, sx + 2, sy + 5], fill=(95, 50, 25))

    elif "Leaf_Blast" in disease:
        # Spindle / diamond shaped lesions with gray center and reddish-brown borders
        for _ in range(random.randint(4, 9)):
            bx = cx + random.randint(-30, 30)
            by = cy + random.randint(-60, 60)
            draw.polygon([
                (bx, by - 16),
                (bx + 8, by),
                (bx, by + 16),
                (bx - 8, by)
            ], fill=(145, 55, 40))
            draw.polygon([
                (bx, by - 8),
                (bx + 4, by),
                (bx, by + 8),
                (bx - 4, by)
            ], fill=(200, 200, 195))

    elif "Bacterial_spot" in disease:
        # Small angular water-soaked dark brown spots with yellow halos
        for _ in range(random.randint(18, 35)):
            bx = cx + random.randint(-55, 55)
            by = cy + random.randint(-55, 55)
            br = random.randint(2, 6)
            draw.ellipse([bx - br - 1, by - br - 1, bx + br + 1, by + br + 1], fill=(195, 185, 30))
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(55, 35, 20))

    elif "Yellow_Leaf_Curl_Virus" in disease:
        # Severe chlorosis and mottled pale yellowing with curled leaf boundaries
        img_np = np.array(img).astype(np.float32)
        # Shift green channel toward yellow (boost red channel)
        mask = (img_np[:, :, 1] > 80)
        img_np[mask, 0] = np.clip(img_np[mask, 0] * 1.9 + 50, 0, 255)
        img_np[mask, 1] = np.clip(img_np[mask, 1] * 1.3 + 30, 0, 255)
        img = Image.fromarray(np.uint8(img_np))

    return img


def generate_sample_dataset(output_dir: str, samples_per_class: int = 35):
    """
    Builds the dataset directory with synthetic high-quality training and validation images.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating balanced dataset in: {output_dir}")

    total_created = 0
    for class_name in CLASSES:
        crop_type = class_name.split("___")[0]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples_per_class):
            base_img = _create_base_leaf(crop_type)
            diseased_img = _apply_disease_symptoms(base_img, class_name)
            
            # Random slight rotation / jitter for data augmentation diversity
            rot = random.uniform(-15, 15)
            final_img = diseased_img.rotate(rot, resample=Image.BILINEAR)

            file_path = os.path.join(class_dir, f"leaf_{i:03d}.jpg")
            final_img.save(file_path, "JPEG", quality=90)
            total_created += 1

    print(f"Successfully generated {total_created} images across {len(CLASSES)} classes.")


def populate_sample_test_images(samples_dir: str):
    """
    Creates a curated gallery of test leaf images for quick 1-click testing in the UI.
    """
    os.makedirs(samples_dir, exist_ok=True)
    sample_classes = [
        "Apple___Apple_scab",
        "Apple___healthy",
        "Corn___Common_rust",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Rice___Leaf_Blast",
        "Tomato___Early_blight",
        "Tomato___Yellow_Leaf_Curl_Virus",
        "Tomato___healthy"
    ]

    for c_name in sample_classes:
        crop = c_name.split("___")[0]
        base_img = _create_base_leaf(crop, width=300, height=300)
        leaf_img = _apply_disease_symptoms(base_img, c_name)
        out_path = os.path.join(samples_dir, f"{c_name}_sample.jpg")
        leaf_img.save(out_path, "JPEG", quality=95)

    print(f"Sample test gallery ready in {samples_dir}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "data", "dataset")
    samples_dir = os.path.join(base_dir, "data", "sample_test_images")

    generate_sample_dataset(dataset_dir, samples_per_class=40)
    populate_sample_test_images(samples_dir)
