# Brain Tumor Detection with VGG-16 Model

## Table of Contents

- [1. Project Overview and Objectives](#intro) 
    - [1.1. Data Set Description](#dataset) 
    - [1.2. What is Brain Tumor?](#tumor)
- [2. Setting up the Environment](#env)
- [3. Data Import and Preprocessing](#import)
- [4. CNN Model](#cnn)
    - [4.1. Data Augmentation](#aug)
        - [4.1.1. Demo](#demo)
        - [4.1.2. Apply](#apply)
    - [4.2. Model Building](#build)
    - [4.3. Model Performance](#perf)
- [5. Conclusions](#concl)

## 1. Project Overview and Objectives

The main purpose of this project was to build a Convolutional Neural Network (CNN) model to classify whether a subject has a brain tumor or not based on MRI scans. I used the [VGG-16](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) model architecture and pretrained weights to train the model for this binary classification problem. 

The model's performance is evaluated based on accuracy, which is defined as:

$\textrm{Accuracy} = \frac{\textrm{Number of correctly predicted images}}{\textrm{Total number of tested images}} \times 100\%$

The final results are as follows:

| Set             | Accuracy |
|-----------------|----------|
| Validation Set  | ~88%     |
| Test Set        | ~80%     |

*Note*: 
- **`Validation set`** - used during model training to adjust the hyperparameters. 
- **`Test set`** - a small set of data not used in the training process, used for final model evaluation.

### 1.1. Data Set Description

The dataset used for this project is [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection). It contains MRI scans belonging to two classes:

- **`NO`** (no tumor) - encoded as `0`
- **`YES`** (tumor) - encoded as `1`

Unfortunately, there is no additional information available about the source of these MRI scans.

### 1.2. What is Brain Tumor?

A brain tumor occurs when abnormal cells form within the brain. There are two main types of brain tumors:
- **Cancerous (Malignant)** - Cancerous tumors can be either primary (starting in the brain) or secondary (spreading from another part of the body).
- **Benign** - Non-cancerous tumors that do not spread to other areas.

Symptoms of a brain tumor can include headaches, seizures, vision problems, vomiting, and changes in mental state, which worsen over time.

![Brain Tumor Example](https://upload.wikimedia.org/wikipedia/commons/5/5f/Hirnmetastase_MRT-T1_KM.jpg)

## 2. Setting up the Environment

First, install the required libraries:

```bash
!pip install imutils
```
Then import the necessary packages for the project:
```
python
import numpy as np
import cv2
import os
import shutil
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
```

### 3. Data Import and Preprocessing
* To work with the dataset, the first step is to organize the images into training, validation, and test sets.
```
python
IMG_PATH = '../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/'
```

* The following code splits the images into training, validation, and test directories:
```
bash
!mkdir TRAIN TEST VAL TRAIN/YES TRAIN/NO TEST/YES TEST/NO VAL/YES VAL/NO
```

3.1. Load Data
* The load_data function loads and resizes images from the specified directory. Additionally, it creates the labels for the images.
```
python
def load_data(dir_path, img_size=(100,100)):
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    return X, y, labels
```

3.2. Crop Brain from Images
* Using OpenCV, we can crop out the brain region from the MRI scans to remove unnecessary background. This improves model performance.
  
```
python
def crop_imgs(set_name, add_pixels_value=0):
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold and crop the brain region
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)
```

3.3. Preprocess the Images
* We resize the images to (224, 224) and apply preprocessing for VGG-16:

```
python
def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)
```

### 4. CNN Model
4.1. Data Augmentation
* Since the dataset is small, I used data augmentation techniques to artificially increase the size of the training set. This includes random rotation, horizontal and vertical flips, and brightness adjustment.

4.1.1. Demo
* Here is an example of how an image is augmented:

```python
demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)
```

4.1.2. Apply
* For training, I applied augmentation using the following code:

```python
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)
```
4.2. Model Building
* I used Transfer Learning with the pretrained VGG-16 model, modifying the final layers to output a binary classification.

```python
vgg16_weight_path = 'path_to_vgg16_weights.h5'
base_model = VGG16(weights=vgg16_weight_path, include_top=False, input_shape=IMG_SIZE + (3,))
model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.layers[0].trainable = False
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
```

4.3. Model Performance
* The model was trained for 30 epochs with early stopping to avoid overfitting. The accuracy and loss for both training and validation sets were plotted.
```python

history = model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=25,
    callbacks=[EarlyStopping(monitor='val_acc', mode='max', patience=6)]
)
```
After training, the model was evaluated on the validation and test sets, achieving an accuracy of approximately 88% on the validation set and 80% on the test set.

### 5. Conclusions
* This project successfully built a CNN model to detect brain tumors from MRI scans. The VGG-16 architecture and transfer learning helped achieve good performance despite a relatively small dataset.
* Further improvements can be made by increasing the dataset size, hyperparameter tuning, or experimenting with other CNN architectures.

### Saving the Model
* The model is saved for future use:
```python
model.save('brain_tumor_detection_vgg16_model.h5')
```

### Clean-up
* To free up space after the process:

```bash
!rm -rf TRAIN TEST VAL TRAIN_CROP TEST_CROP VAL_CROP
```
---
### Why VGG-16?
The VGG-16 model was chosen for this project due to several important reasons related to its architecture and performance characteristics. Here’s why VGG-16 is a good choice for tasks like brain tumor detection in MRI scans:

1. Pretrained on Large Datasets (ImageNet)
* VGG-16 is a Convolutional Neural Network (CNN) that was pretrained on the ImageNet dataset, which consists of millions of labeled images from thousands of classes.
* This means that VGG-16 already has learned useful features (e.g., edges, textures, shapes) from a large and diverse dataset. When fine-tuned on a smaller dataset, like the brain tumor detection dataset, the model can leverage these learned features for better generalization, which is known as Transfer Learning.

2. Depth of the Network
* VGG-16 has 16 layers — hence the name "VGG-16" — which makes it deep enough to capture complex patterns and features in images.
* The architecture consists of 13 convolutional layers and 3 fully connected layers, enabling it to learn hierarchical features from the input image (e.g., low-level features like edges in earlier layers and higher-level features like faces, animals, or objects in deeper layers). For tasks like tumor detection, capturing high-level features in MRI scans can improve classification performance.

3. Simple and Consistent Architecture
* The VGG-16 model is widely used due to its simplicity and uniform structure.
* All convolution layers use 3x3 filters with a stride of 1 and pooling layers use 2x2 max-pooling, making it easy to implement, modify, and understand.
* Its consistent architecture allows researchers to easily experiment with slight modifications to improve performance or adapt it to different tasks.

4. Good Performance on Vision Tasks
* VGG-16 has demonstrated strong performance on a variety of image recognition and classification tasks.
* Despite being an older model (compared to more recent architectures like ResNet or EfficientNet), it is still a reliable choice for transfer learning and has become a popular baseline in many computer vision applications.

5. Availability of Pretrained Weights
* Since VGG-16 has been extensively used in the research community, pretrained weights are easily accessible.
* These weights, when loaded into the model, allow the network to start from a well-optimized configuration, reducing the training time significantly.
* For small datasets, like MRI images for tumor detection, starting with these pretrained weights provides a good starting point, often yielding better results than training a model from scratch.

6. Model Simplicity and Computational Efficiency
* While VGG-16 is deep, it does not have extremely complex components or operations like residual connections (as in ResNet).
* This simplicity makes VGG-16 relatively computationally efficient and easy to train on moderate hardware (e.g., a good GPU), which is beneficial if you have limited computational resources.









