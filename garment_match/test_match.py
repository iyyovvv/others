from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# Load FashionCLIP Model
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")  # Replace with actual model name
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")  # Replace if different
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Input Image
input_image_path = "input.jpg"  # Your input image
input_image = Image.open(input_image_path)

# Folder of Garment Images
garment_folder = "dress"  # Your folder of images
garment_paths = [os.path.join(garment_folder, f) for f in os.listdir(garment_folder) if f.endswith((".jpg", ".png"))]

# Preprocess Images
input_image_processed = processor(images=input_image, return_tensors="pt").to(device)
garment_images_processed = [processor(images=Image.open(path), return_tensors="pt").to(device) for path in garment_paths]

# Encode Images
with torch.no_grad():
    input_features = model.get_image_features(**input_image_processed)
    garment_features = torch.cat([model.get_image_features(**g) for g in garment_images_processed], dim=0)

# Normalize Features
input_features = input_features / input_features.norm(dim=-1, keepdim=True)
garment_features = garment_features / garment_features.norm(dim=-1, keepdim=True)

# Compute Similarity Scores
similarities = (garment_features @ input_features.T).squeeze(1).cpu().numpy()

# Sort and Find Top Matches
top_k = 5
best_indices = np.argsort(similarities)[::-1][:top_k]
print("Top Matches:")
for idx in best_indices:
    print(f"{garment_paths[idx]} (Similarity: {similarities[idx]:.4f})")

# Display Results
def display_results(input_image_path, garment_paths, best_indices):
    input_img = Image.open(input_image_path)
    garment_imgs = [Image.open(garment_paths[idx]) for idx in best_indices]

    plt.figure(figsize=(15, 5))
    
    # Show input image
    plt.subplot(1, len(garment_imgs) + 1, 1)
    plt.imshow(input_img)
    plt.axis("off")
    plt.title("Input Image")
    
    # Show top matches
    for i, garment_img in enumerate(garment_imgs):
        plt.subplot(1, len(garment_imgs) + 1, i + 2)
        plt.imshow(garment_img)
        plt.axis("off")
        plt.title(f"Match {i + 1}")
    
    plt.show()

# Visualize results
display_results(input_image_path, garment_paths, best_indices)
