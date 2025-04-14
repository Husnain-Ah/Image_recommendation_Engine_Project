import os
import numpy as np
import tensorflow as tf
from annoy import AnnoyIndex
from PIL import Image
import json

BASE_DIR = 'C:/Users/Husnain/Desktop/git for uni work/Image_recommendation_Engine_Project/tiny-imagenet-200/'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
WORDS_FILE = os.path.join(BASE_DIR, 'words.txt')

FEATURE_DIM = 1280
TOP_K = 10
MAX_IMAGES_PER_CLASS = 5 # only load 5 pics for now to cut dowen load time in developmnet


label_map = {}
with open(WORDS_FILE, 'r') as f:
    for line in f:
        wnid, label = line.strip().split('\t')
        label_map[wnid] = label

model = tf.keras.Sequential([
    tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
    tf.keras.layers.GlobalAveragePooling2D()
])
model.trainable = False


def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).resize(target_size).convert("RGB")
    img = np.array(img) / 255.0
    return img

def extract_embeddings(image_paths):
    embeddings = []
    metadata = []

    for img_path in image_paths:
        img = preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        embedding = model.predict(img, verbose=0).flatten()

        parts = img_path.split(os.sep)
        wnid = parts[-3]  
        filename = parts[-1]
        label = label_map.get(wnid, wnid)

        embeddings.append(embedding)
        metadata.append({
            "filename": filename,
            "path": img_path,
            "label": label,
            "wnid": wnid
        })

    return np.array(embeddings), metadata

image_paths = []
for wnid in os.listdir(TRAIN_DIR):
    wnid_dir = os.path.join(TRAIN_DIR, wnid, 'images')
    if not os.path.isdir(wnid_dir):
        continue
    image_files = [f for f in os.listdir(wnid_dir) if f.endswith('.JPEG')][:MAX_IMAGES_PER_CLASS]
    for file in image_files:
        image_paths.append(os.path.join(wnid_dir, file))

print(f"Processing {len(image_paths)} images...")
embeddings, metadata = extract_embeddings(image_paths)

annoy_index = AnnoyIndex(FEATURE_DIM, 'angular')
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)

annoy_index.build(10)
annoy_index.save("annoy_data/image_annoy_index.ann")

neighbors = {}
for i in range(len(embeddings)):
    indices = annoy_index.get_nns_by_item(i, TOP_K + 1)
    current_file = metadata[i]['filename']
    neighbors[current_file] = [
        metadata[idx]['filename'] for idx in indices if metadata[idx]['filename'] != current_file
    ]

with open('annoy_data/annoy_neighbors.json', 'w') as f:
    json.dump(neighbors, f, indent=2)

with open('annoy_data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Annoy index and metadata saved successfully.")
