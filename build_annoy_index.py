import os
import numpy as np
import tensorflow as tf
from annoy import AnnoyIndex
from PIL import Image
import json

IMAGE_DIR = 'C:/Users/Husnain/Desktop/git for uni work/Image_recommendation_Engine_Project/tiny-imagenet-200/train/'

FEATURE_DIM = 1280  
TOP_K = 10  

model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
model.trainable = False  

def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).resize(target_size)
    img = np.array(img) / 255.0 
    return img

def extract_embeddings(image_paths):
    embeddings = []
    filenames = []
    
    for img_path in image_paths:
        img = preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)  
        embedding = model.predict(img)  
        embedding = embedding.flatten()  
        embeddings.append(embedding)
        filenames.append(os.path.basename(img_path))
    
    return np.array(embeddings), filenames

image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]

embeddings, filenames = extract_embeddings(image_paths)

annoy_index = AnnoyIndex(FEATURE_DIM, 'angular')  

for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)


annoy_index.build(10)

annoy_index.save("image_annoy_index.ann")

neighbors = {}
for i in range(len(embeddings)):
    neighbor_indices = annoy_index.get_nns_by_item(i, TOP_K + 1)  
    neighbors[filenames[i]] = [filenames[idx] for idx in neighbor_indices if filenames[idx] != filenames[i]]

with open('annoy_neighbors.json', 'w') as f:
    json.dump(neighbors, f)

print("Annoy index and neighbors saved successfully.")
