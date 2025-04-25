import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

def visualize_infection(image_path, save_path="static/infected_result.jpg"):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Step 1: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)

    # Step 2: K-means clustering
    Z = enhanced_img.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    segmented = labels.reshape(img.shape)

    # Step 3: Create binary mask from the cluster with highest intensity
    infected_mask = np.uint8(segmented == segmented.max()) * 255

    # Step 4: Morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    infected_mask = cv2.morphologyEx(infected_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 5: Overlay on original image
    heatmap = cv2.applyColorMap(infected_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.7, heatmap, 0.3, 0)

    # Save the result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)

    return save_path
# Load the segmentation model
segmentation_model = load_model('models/CancerSegmentationModel.h5')

def display_predicted_mask(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (256, 256))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)

    # Predict mask
    pred_mask = segmentation_model.predict(img_input)[0].squeeze()

    # Create color overlay mask
    mask_colored = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR), 0.6, mask_colored, 0.4, 0)

    # Save the overlay image
    filename = os.path.basename(image_path)
    save_path = os.path.join('static', 'uploads', f"mask_{filename}")
    cv2.imwrite(save_path, overlay)

    return save_path  # Return the path so Flask can show it