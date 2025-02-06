## GtiHub repo

https://github.com/achgue/Object-Detection-with-YOLOv3

# YOLOv3 Object Detection with Keras

This repository provides an implementation of YOLOv3 (You Only Look Once version 3) for object detection using Keras and TensorFlow.  
It detects objects in images using a pre-trained YOLOv3 model trained on the COCO dataset.  
The script processes images in a folder and outputs images with detected objects and their corresponding bounding boxes.  
The detection threshold is set to **55%**, meaning only predictions with a confidence score above 55% will be considered.

## Steps to Execute:

1. **Prepare Input Images**

   - Save your images in the `input_images` folder.
   - Images should follow the format: `image_n.jpg` (e.g., `image_1.jpg`, `image_2.jpg`, ...).

2. **Navigate to the Project Root**

   - Open a terminal/console and move to the project's root folder:

   ```bash
   cd path/to/Object-Detection-with-YOLOv3
   ```

3. **Run the Detection Script**

   - Execute the following command:

   ```bash
   python yolo_detection.py
   ```

4. **Check Results**
   - The YOLOv3 model is saved in the `models` folder.
   - Detection results (processed images) are saved in the `output_images` folder.
