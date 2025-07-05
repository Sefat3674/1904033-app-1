



# --- Install required packages (run in terminal or notebook cell) ---
#!pip install tensorflow==2.11.0 --force-reinstall
#!pip install --upgrade keras-cv-attention-models


# --- Imports ---
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from keras_cv_attention_models.coatnet import CoAtNet0
from sklearn.model_selection import train_test_split
from skimage.segmentation import mark_boundaries
from lime import lime_image
import tensorflow as tf

# --- Constants ---
IMG_SIZE = (224, 224)
NUM_CLASSES = 8

# --- Load or create the model ---
model = CoAtNet0(pretrained="imagenet", input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# --- Your preprocessing functions and pipeline below ---
def crop_circle(img):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist <= radius
    if img.ndim == 3:
        mask = np.stack([mask] * 3, axis=-1)
    img[~mask] = 0
    return img

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def sharpen_image(img, sigma=10):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)

def resize_normalize(img, size=(224, 224)):
    img = cv2.resize(img, size)
    img = img / 255.0
    return img

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_circle(img)
    img = apply_clahe(img)
    img = sharpen_image(img)
    img = resize_normalize(img)
    return img.astype(np.float32)



# --- Load Labels CSV ---
test_images_folder = '/kaggle/input/test-image/saved_test_images'
labels_csv_path = '/kaggle/input/test-image/saved_test_labels.csv'

labels_df = pd.read_csv(labels_csv_path)
filename_to_label_idx = dict(zip(labels_df['filename'], labels_df['label']))
label_to_classname = dict(zip(labels_df['label'], labels_df['class_name']))

def idx_to_class(idx):
    return label_to_classname.get(idx, "Unknown")

# --- Load Model ---
model_path = "/kaggle/input/92.11/tensorflow2/default/1/best_model_finetuned(0.9211).keras"
model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Find last conv layer ---
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'mhsa_output' in layer.name:
            return layer.name
    raise ValueError("No suitable conv layer found.")
last_conv_layer_name = find_last_conv_layer(model)

# --- Grad-CAM ---
def generate_gradcam(model, img_array, class_index, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- LIME Explainer ---
explainer = lime_image.LimeImageExplainer()
def predict_fn(images): 
    return model.predict(np.array(images), verbose=0)

# --- Explanation text ---
explanation_text = {
    'Normal': "Model predicted Normal based on healthy optic disc and macula.",
    'Diabetes': "Detected retinal blood vessel changes suggestive of Diabetes.",
    'Glaucoma': "Detected increased cupping in the optic disc indicating Glaucoma.",
    'Cataract': "Image blur indicated potential Cataract.",
    'AMD': "Degeneration signs in macula indicate AMD.",
    'Hypertension': "Blood vessel narrowing/hemorrhages indicate Hypertension.",
    'Myopia': "Tilted disc and fundus shape suggest Myopia.",
    'Others': "Non-specific features detected, marked as Others."
}

# --- Visualization functions ---
def visualize_preprocessing_pipeline(img_path):
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"Image not found: {img_path}")
        return
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    imgs = {'original': orig_img}
    img = crop_circle(orig_img.copy())
    imgs['circle'] = img.copy()
    img = apply_clahe(img.copy())
    imgs['clahe'] = img.copy()
    img = sharpen_image(img.copy())
    imgs['sharpen'] = img.copy()
    img = resize_normalize(img.copy())
    img = (img * 255).astype(np.uint8)
    imgs['resized'] = img.copy()

    steps_to_show = ['original', 'circle', 'clahe', 'sharpen', 'resized']
    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(20, 4))
    for i, step in enumerate(steps_to_show):
        axes[i].imshow(imgs[step])
        axes[i].set_title(step.capitalize())
        axes[i].axis('off')
    plt.suptitle(f"Preprocessing Stages for: {os.path.basename(img_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()

def display_combined_visualization(img_path, true_label, pred_label, pred_idx, layer_name):
    img_array = preprocess_image(img_path)
    if img_array is None:
        return
    input_array = np.expand_dims(img_array, axis=0)

    # Grad-CAM
    heatmap = generate_gradcam(model, input_array, pred_idx, layer_name)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
    heatmap_rgb = cm.jet(heatmap / 255.0)[..., :3]
    heatmap_rgb = np.uint8(heatmap_rgb * 255)
    overlayed = cv2.addWeighted(np.uint8(img_array * 255), 0.5, heatmap_rgb, 0.5, 0)

    # LIME
    explanation = explainer.explain_instance(
        image=img_array, classifier_fn=predict_fn,
        top_labels=1, hide_color=0, num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(label=pred_idx, positive_only=True, num_features=10, hide_rest=False)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_array)
    axs[0].set_title(f"Original\nTrue: {true_label}", fontsize=11)
    axs[1].imshow(overlayed)
    axs[1].set_title(f"Grad-CAM\nPred: {pred_label}", fontsize=11)
    axs[2].imshow(mark_boundaries(temp, mask))
    axs[2].set_title(f"LIME\nPred: {pred_label}", fontsize=11)
    for ax in axs: ax.axis('off')
    summary = explanation_text.get(pred_label, "Model detected features matching this class.")
    plt.figtext(0.5, 0.01, summary, wrap=True, ha='center', fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    plt.close()

# --- Main Pipeline ---

# List all test images in the folder
all_test_images = [os.path.join(test_images_folder, f) for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Pick 5 random images (or less if folder has fewer)
np.random.seed(42)
sampled_images = np.random.choice(all_test_images, size=min(5, len(all_test_images)), replace=False)

for img_path in sampled_images:
    filename = os.path.basename(img_path)
    true_label_idx = filename_to_label_idx.get(filename, None)
    true_label = idx_to_class(true_label_idx) if true_label_idx is not None else "Unknown"
    print(f"\n📸 Processing image: {filename}")
    print(f"✅ True Label: {true_label}")

    # Show preprocessing steps
    visualize_preprocessing_pipeline(img_path)

    # Predict
    img_array = preprocess_image(img_path)
    if img_array is None:
        print(f"Skipping {filename} due to read error.")
        continue
    input_array = np.expand_dims(img_array, axis=0)
    pred_probs = model.predict(input_array, verbose=0)
    pred_idx = np.argmax(pred_probs)
    pred_label = idx_to_class(pred_idx)

    print(f"🔮 Predicted Label: {pred_label}")

    # Show explanations
    display_combined_visualization(img_path, true_label, pred_label, pred_idx, last_conv_layer_name)
