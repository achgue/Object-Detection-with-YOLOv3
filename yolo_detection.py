import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils import download_file
from src.yolo import YOLO3, decode_predictions, draw_boxes

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Download YOLOv3 model weights
yolo_model_url = "https://jeffpro.blob.core.windows.net/public/coco_yolo3.h5"
model_filename = "models/coco_yolo3.h5"
download_file(yolo_model_url, model_filename)

# Load the YOLO model
model = load_model(model_filename)
print("Model loaded successfully.")

# Ensure the 'images' folder exists
image_folder = "input_images"
os.makedirs(image_folder, exist_ok=True)

# Iterate through images in the folder
for i in range(1, 11):  # Images named from images_1.jpg to images_10.jpg
    image_path = os.path.join(image_folder, f"image_{i}.jpg")

    if not os.path.exists(image_path):
        print(f"Skipping {image_path} (not found).")
        continue  # Skip missing images

    print(f"Processing {image_path}...")

    # Preprocess image
    #image = preprocess_image(image_path)
    image = plt.imread(image_path)
    width, height = image.shape[1], image.shape[0]
    
    x = load_img(image_path, target_size=(YOLO3.width, YOLO3.height))
    x = img_to_array(x) / 255
    x = np.expand_dims(x, axis=0)
    y = model.predict(x)

    # Decode predictions into bounding boxes (implement this function)
    boxes = decode_predictions(y, width, height, min_score=0.55)

    for box in boxes:
      print(f'({box.xmin}, {box.ymin}), ({box.xmax}, {box.ymax}), {box.label}, {box.score}')

    # Draw and display bounding boxes
    draw_boxes(image_path, boxes)
