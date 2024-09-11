## Table of Contents

1. [Step 1: Export images from QuPath and setup directories](#step-1-export-images-from-qupath-and-setup-directories)
2. [Step 2: Setup environment to run YOLO model](#step-2-setup-environment-to-run-yolo-model)
3. [Step 3: Convert image masks to labels](#step-3-convert-image-masks-to-labels)
4. [Step 4: Validate created labels through visualization](#step-4-validate-created-labels-through-visualization)
5. [Step 5: Train model](#step-5-train-model)
6. [Step 6: Validate trained model](#step-6-validate-trained-model)
7. [Step 7: Repeat steps 5-6 with different versions of YOLO model and parameters until the requirements are met](#step-7-repeat-steps-5-6-with-different-versions-of-yolo-model-and-parameters-until-the-requirements-are-met)
8. [Findings and Recommendations](#findings-and-recommendations)

---------------------------------------------------------------------------------
### Step 1: Export images from QuPath and setup directories:
---------------------------------------------------------------------------------

- Export images from QuPath Using 'export_image_annotations_v2.groovy' script located in this folder.
- Once all images are exported:
    - Split the images into training and test sets, and move the images into 'train' and 'val' (for test) folders.
    - Inside both 'train' and 'val' folders, create three new folders:
        - images
        - masks
        - labels (will be empty for now)

    - Move the images (.jpg) into 'images' folder.
    - Move the mask (.png) files into 'masks' folder.
    - Folder 'labels' will be empty for now. Labels will be create in the next step.
    
- Directory file structure should look like the following:
    - root:
        - main.ipynb (this file)
        - datasets
            - seg
                - train
                    - images
                        - 1.jpg
                    - masks
                        - 1.png
                    - labels
                        - 1.txt (will be created in the next step)
                - val
                    - images
                        - 2.jpg
                    - masks
                        - 2.png
                    - labels
                        - 2.txt (will be created in the next step)

- Now, we are ready to move on to the next step.


---------------------------------------------------------------------------------
### Step 2: Setup environment to run YOLO model:
---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
#### A: Install required packages to run this notebook:
---------------------------------------------------------------------------------


```python
pip install -r requirements.txt
```

---------------------------------------------------------------------------------
#### B: Create config file:
---------------------------------------------------------------------------------


```python
# Create config file 'config.yaml' with following contents [Remove Hash]:

#path: ./seg/
#train: train/images/
#val: val/images/

#nc: 1  # number of classes
#names: [ 'tumor']
```

### Step 3: Convert image masks to labels:

---------------------------------------------------------------------------------
#### A: Setup Functions to convert masked images to YOLO segmentation format:
---------------------------------------------------------------------------------

Functions:
1. find_polygons_in_binary_mask(...) - Finds and returns a polygon in a masked image
3. parse_segmentation_to_yolo_format(...) - Converts a masked image to YOLO segementation label txt file

----------------------------------------------------------------


```python
import numpy as np
import cv2

def find_polygons_in_binary_mask(binary_mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = [polygon.squeeze() for polygon in contours]
    return polygons

def parse_segmentation_to_yolo_format(mask: np.ndarray, background: int | None = None, contour: int | None = None) -> list[list[int | float]]:
    """ Return list of YOLO like annotations for segmentation purposes, i.e. 
    [
        [class_id, x1, y1, x2, y2, ..., xn, yn], # object_0
        [class_id, x1, y1, x2, y2, ..., xn, yn], # object_1
        ...,
        [class_id, x1, y1, x2, y2, ..., xn, yn] # object_m
    ]
    """
    unique_label_ids = np.unique(mask).tolist()
    if background is not None:
        unique_label_ids.remove(background) # remove background from labels
    if contour is not None:
        unique_label_ids.remove(contour) # remove contour from labels
    annotations = []
    for label_id in unique_label_ids:
        id = 0
        binary_mask = ((mask == label_id) * 1).astype(np.uint8)
        polygons = find_polygons_in_binary_mask(binary_mask)
        wh = np.flip(np.array(binary_mask.shape)) # for normalization purposes
        norm_polygons = [polygon / wh for polygon in polygons]
        xy_sequences = [polygon.flatten().tolist() for polygon in norm_polygons]
        for xy_sequence in xy_sequences:
            annotations.append([id] + xy_sequence)
    return annotations

```

---------------------------------------------------------------------------------
#### B: Start conversion of masked images to YOLO segmentation label file:
---------------------------------------------------------------------------------


```python
import numpy as np
from PIL import Image

def process_mask(mask_file, mask_path):
    raw_mask = Image.open(mask_image)
    mask = np.array(raw_mask)
    print(mask_file)
    annotations = parse_segmentation_to_yolo_format(mask, background=0, contour=None)
    output_label_path = os.path.splitext(os.path.basename(mask_file))[0] + '.txt'
    if not os.path.exists(mask_path + "labels/"):
        os.mkdir(mask_path + "labels/")
    with open(mask_path + "labels/" + output_label_path, "w") as f:
        f.write(' '.join(str(x) for x in annotations[0]))

if __name__ == "__main__":
    for keyword in ["train", "val"]:
        mask_path = "datasets/seg/{}/".format(keyword)
        for mask_image in glob.glob(mask_path + "masks/" + "*.png"):
            process_mask(mask_image, mask_path)
```

---------------------------------------------------------------------------------
### Step 4: Validate created labels through visualization:
---------------------------------------------------------------------------------


```python
import cv2
import numpy as np

img_path0 = "datasets/seg/train/"
img_name = "NP_MIT_0004_A4.27 [d=1.76228,x=38573,y=36543,w=1128,h=1127]"

image = cv2.imread(img_path0 + "images/" + img_name + ".jpg")

with open(img_path0 + "labels/" + img_name + ".txt") as f:
    segment = [np.array(x.split(), dtype=np.float32)[1:].reshape(-1, 2) for x in f.read().strip().splitlines() if len(x)]
h, w = image.shape[:2]
for s in segment:
    s[:, 0] *= w
    s[:, 1] *= h
cv2.drawContours(image, [s.astype(np.int32) for s in segment], -1, (0, 0, 255), thickness=2)
cv2.imwrite("1.jpg", image)
```

---------------------------------------------------------------------------------
### Step 5: Train model:
---------------------------------------------------------------------------------


```python
from ultralytics import YOLO

# Train a custom model
model = YOLO("yolov8n-seg.pt")
model.train(data="data_seg_config.yaml", epochs=100, imgsz=640, batch=4, close_mosaic=0, project='bd', flipud=0.5, mosaic=0.5)

# OR

model = YOLO("yolov8n.pt")
model.train(data="data_seg_config.yaml", epochs=300, imgsz=640)
```

---------------------------------------------------------------------------------
### Step 6: Validate trained model:
---------------------------------------------------------------------------------

---------------------------------------------------------------------------------
#### Validate using predict function:
---------------------------------------------------------------------------------


```python
# from ultralytics import YOLO

trainedModel = YOLO("runs/segment/train/weights/best.pt")
trainedModel.predict(source="image.png", save=False, imgsz=[4000,3000]) #save_txt=True) #conf=0.5
```

---------------------------------------------------------------------------------
#### Validate using SAHI inference:
---------------------------------------------------------------------------------


```python
from sahi.utils.file import download_from_url

base_path = "predict/"
yolov8_model_path = base_path + "input/trained_model.pt"
img_path = base_path + "input/image_to_predict.jpg"
```


```python
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.1,
    device="cpu",  # or 'cuda:0'
)
```


```python
from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    img_path,
    detection_model,
    slice_height=128,
    slice_width=128,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
```


```python
result.export_visuals(export_dir=base_path + "output/")
Image.open(base_path + "output/predicted_image.png")
```

---------------------------------------------------------------------------------------------
### Step 7: Repeat steps 5-6 with different versions of YOLO model and parameters until the requirements are met.
--------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------
### Findings and Recommendations:
--------------------------------------------------------------------------------------------

In Step 1, we exported the images using the default resolution of 1960 x 1200 as specified in the script. We encountered an issue with exporting annotated images because they were not categorized into Tumor or other classifications. Consequently, the export script generated zero annotated images. To resolve this, we classified all annotations as Tumor in QuPath and saved the images before re-running the export script. This adjustment produced the expected annotated images and masks.

In Step 2, after setting up the necessary environments, Step 3 involved converting the image masks (.png) into YOLO label format (.txt) files. We experimented with various conversion scripts and selected the most effective one. Validation of the labels, as detailed in Step 4, was crucial.

After validating the images and labels, we proceeded to model training in Step 5. We tested different hyperparameters and YOLOv8 versions (v8n, v8x, v8n-seg, v8n-p2, etc.). Initially, the model failed to detect cells effectively. However, after numerous trials, we discovered that lowering the image resolution in Step 1 and adjusting the epochs to 100-300 improved accuracy and detection rates. Due to limited computing resources, we used the paid version of Google Colab (Pro) for training.

Once the model was trained, we validated its performance using the prediction function. Although it detected some cells, it missed others. To enhance detection, we applied SAHI inference (Step 6), which improves detection by dividing the image into smaller sections, processing each section, and then reconstructing the full image with enhanced predictions.

Repeat Steps 5 and 6 with different parameters and models as needed until deployment readiness. You can find some of our recent results on the trained models in 'result-yolov8n-320-300epoch' (trained on image size of 320 x 320 for 300 epochs) and 'result-yolov8-seg-640-300epoch' (trained on image size of 640 x 640 for 300 epochs) folders. Due to time constraints and an incomplete model, we could not progress further. To improve the model, consider the following recommendations:

- Increase Input Data: [Ultralytics recommends using at least 1,500 images and 10,000 annotations per class.](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)

- Explore Newer Models: Test newer YOLO versions (v9, v10) and other models like Mask R-CNN or RetinaNet.
- Experiment With Input Data: Experiment Lower the pixels/resolution of the images when exporting them on Step 1 (for example: 640 x 640). Lowering image resolutions increased the detection rate in our case. Results are in 'result...' folders in this repository with best trained model.
- Optimize Conversion Functions: Similarly, consider optimizing the conversion function from image masks to YOLO label formt on Step 3.
- Use Polygon Annotations: Consider models compatible with polygon-shaped annotations for improved accuracy. Mask R-CNN supports polygons and may offer better performance than YOLO, which converts polygons into rectangles.
- Utilize Cloud Services: For more computing power, use platforms like Google Colab Pro or Kaggle, which can significantly speed up the training process, especially under tight deadlines.


```python

```
