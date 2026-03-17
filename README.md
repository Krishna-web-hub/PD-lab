# Food Image Preprocessing & Data Pipeline

## Project Goal

Build a machine learning pipeline that can analyze food images and help detect:

* **AI-generated food images**
* **Fresh vs Spoiled food**
* **Food contamination (hair, insects, plastic, etc.)**

The current work focuses on preparing the **dataset and data pipeline** for the **Fresh vs Spoiled food classification task**.

---

# 1. Dataset Collection

Images were collected and organized into three initial folders:

```
Images/
   AI_gen_food/
   good_food/
   bad_food/
```

Meaning:

| Folder      | Meaning                  |
| ----------- | ------------------------ |
| AI_gen_food | AI-generated food images |
| good_food   | Fresh food images        |
| bad_food    | Spoiled food images      |

---

# 2. Image Preprocessing

A preprocessing script was created to standardize the dataset.

### Operations performed

1. Load images using OpenCV
2. Remove corrupted images
3. Resize images to **224 × 224**
4. Save processed images to a new folder

### Resulting structure

```
processed_images/
   AI_gen_food/
   good_food/
   bad_food/
```

Purpose:

* Ensure all images have the same dimensions
* Make them compatible with deep learning models

---

# 3. Dataset Organization by Task

Since the project has multiple AI tasks, the dataset was reorganized.

New structure:

```
dataset/

ai_detection/
   ai_generated/
   real_food/

spoilage_detection/
   fresh/
   spoiled/
```

Mapping:

| Original Folder | New Label         |
| --------------- | ----------------- |
| AI_gen_food     | ai_generated      |
| good_food       | real_food / fresh |
| bad_food        | spoiled           |

---

# 4. Train / Validation / Test Split

To properly evaluate model performance, the dataset was split into:

```
dataset/

train/
val/
test/
```

Split ratio:

```
70% training
15% validation
15% testing
```

Example:

```
dataset/train/spoilage_detection/fresh
dataset/train/spoilage_detection/spoiled
```

---

# 5. Image Tensor Conversion

Images must be converted into numerical tensors before they can be processed by neural networks.

Using **PyTorch transforms**:

```
Resize(224,224)
ToTensor()
```

The transformation converts:

```
Image → Tensor
224 × 224 × 3 → 3 × 224 × 224
```

Pixel values are normalized from:

```
0–255 → 0–1
```

---

# 6. Data Loading with DataLoader

The dataset was loaded using PyTorch utilities.

Components used:

* **ImageFolder** – reads images and labels from folder structure
* **DataLoader** – loads images in batches

Example batch shape:

```
torch.Size([16, 3, 224, 224])
```

Meaning:

```
16 images per batch
3 color channels
224×224 resolution
```

---

# 7. Dataset Verification

Before training the model, the pipeline was verified by:

1. Loading image batches
2. Converting tensors back to images
3. Visualizing multiple images with labels

Example output:

```
fresh | spoiled | fresh
spoiled | fresh | spoiled
...
```

Purpose:

* Ensure labels match images
* Verify preprocessing pipeline
* Confirm DataLoader works correctly

---

# 8. Current Pipeline Status

The completed pipeline so far:

```
Raw Images
    ↓
Preprocessing (resize & clean)
    ↓
Dataset organization
    ↓
Train / validation / test split
    ↓
Tensor conversion
    ↓
DataLoader batching
    ↓
Dataset verification
```

This confirms the dataset is **ready for model training**.

---

# 9. Next Steps

Upcoming work will include:

1. Train a **Fresh vs Spoiled classifier**
2. Implement **AI-generated food detection**
3. Build **contamination detection model**
4. Combine models into a **multi-stage pipeline**

Proposed architecture:

```
Image Input
     ↓
AI Detection
     ↓
Fresh vs Spoiled Classification
     ↓
Contamination Detection
     ↓
Final Decision
```

---

# 10. Tools Used

| Tool        | Purpose                      |
| ----------- | ---------------------------- |
| Python      | Programming                  |
| OpenCV      | Image preprocessing          |
| PyTorch     | Deep learning framework      |
| Torchvision | Dataset loading & transforms |
| Matplotlib  | Visualization                |

---

# Summary

So far, the project has successfully:

* Collected and organized food image datasets
* Built a preprocessing pipeline
* Converted images into tensors
* Implemented PyTorch DataLoader
* Verified dataset correctness through visualization

The system is now **ready for model training and experimentation**.

# Pre-processing Files

All preprocessing scripts are inside the **pre_processing/** folder.

| File | Purpose |
|-----|--------|
| `preprocessing.py` | Reads images, removes corrupted files, and resizes images to **224×224**. Saves them to `processed_images/`. |
| `organise_dataset.py` | Reorganizes the dataset into task folders such as **ai_detection** and **spoilage_detection**. |
| `split_dataset.py` | Splits the dataset into **train**, **validation**, and **test** sets. |
| `tensor_pipeline.py` | Loads images, applies transforms, and converts them to **PyTorch tensors**. |
| `visualize_batch.py` | Displays a batch of images with labels to verify the dataset. |

These scripts together prepare the dataset before model training.
- - -
- - -