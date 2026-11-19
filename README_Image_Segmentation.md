# Image Segmentation Using Attention R2-UNet

## Project Overview

This project implements **binary image segmentation** using PyTorch and a Nature dataset containing images of **butterflies and squirrels**. The goal is to predict pixel-level masks for the objects in the images.

The notebook explores dataset preparation, mask-aware data augmentation, different U-Net architectures, loss functions, and model evaluation. It compares plain, standard-augmentation, and mask-aware crop-augmentation pipelines.

## Dataset

The project uses a custom Nature dataset containing:

- Butterfly images and their segmentation masks
- Squirrel images and their segmentation masks
- JSON annotations that are converted into masks using `create_mask_from_json`

The notebook's dataset exploration notes 300 images and 300 masks for each class, and also mentions an additional set of 100 images. The notebook does not fully clarify the split represented by those additional images, so check the dataset folders before training.

### Dataset Setup

The notebook demonstrates extracting a compressed dataset (for example, `Nature.zip`) into a `Nature` folder. It expects training and test directories to be configured in the notebook as `TRAIN_DIR` and `TEST_DIR`.

Make sure the dataset files, JSON annotations, and required project utility modules are available in the working directory.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- JSON and Python standard libraries
- ResNet34 pretrained encoder (for the pretrained model)

## Project Workflow

1. **Import libraries** – Load PyTorch, Torchvision, NumPy, Pandas, Matplotlib, and supporting modules.
2. **Select compute device** – Use CUDA when a compatible GPU is available; otherwise use CPU.
3. **Prepare the dataset** – Extract the Nature dataset and configure training and test directories.
4. **Explore images and masks** – Inspect class counts, image dimensions, and mask dimensions.
5. **Convert annotations** – Convert JSON annotations into segmentation masks.
6. **Preprocess the data** – Convert images to tensors and prepare masks in a format suitable for PyTorch.
7. **Apply augmentation** – Compare plain data, standard augmentation, and custom mask-aware crop augmentation.
8. **Create DataLoaders** – Prepare training and test DataLoaders for the different pipelines.
9. **Train and compare models** – Evaluate U-Net, Attention U-Net, Attention R2-UNet, and pretrained Attention R2-UNet configurations.
10. **Evaluate segmentation** – Track loss, pixel accuracy, Dice coefficient, IoU, and mean Average Precision (mAP).
11. **Visualize predictions** – Compare input images, ground-truth masks, predicted masks, and segmentation changes across epochs.

## Data Augmentation

The notebook compares three main data pipelines:

- **Plain:** Original images and masks without augmentation.
- **Augmentation-only:** Standard image and mask transformations.
- **Crop + augmentation:** Custom mask-aware cropping followed by compatible geometric and image transformations.

The standard transformations explored include horizontal and vertical flips, rotations, cropping, resizing, and color adjustments.

### Mask-Aware Cropping

A custom `Augmentation.MaskAwareRandomCrop` operation is used to focus on important object regions. The notebook also describes a hard-attention crop with a crop size of 200. This is intended to enlarge small features, such as butterfly antennae, while avoiding cutting through the segmentation mask.

Geometric transformations must be applied consistently to both images and masks. Photometric transformations such as brightness, contrast, saturation, and hue should be applied to images only.

## Models Compared

| Model | Description |
|---|---|
| U-Net | Encoder-decoder architecture with skip connections |
| U-Net++ | Nested U-Net architecture with dense skip connections |
| Attention U-Net | U-Net with attention gates to focus on relevant regions |
| Attention R2-UNet | Attention U-Net with residual recurrent blocks |
| Pretrained Attention R2-UNet | Attention R2-UNet using a ResNet34 encoder |

The notebook's experiments and final evaluation focus on U-Net, Attention U-Net, Attention R2-UNet, and Pretrained Attention R2-UNet.

## Loss Function

The project tests combinations of Binary Cross-Entropy (BCE) loss and Dice loss.

The selected combined loss is:

\[
\text{Loss} = 0.7 \times \text{DiceLoss} + 0.3 \times \text{BCELoss}
\]

Dice loss receives a weight of 0.7 because mask overlap is a primary objective, particularly for a small dataset.

## Evaluation Metrics

- **Pixel Accuracy:** Percentage of pixels classified correctly.
- **Dice Coefficient:** Measures overlap between predicted and ground-truth masks.
- **Intersection over Union (IoU):** Measures the intersection divided by the union of predicted and ground-truth regions.
- **Mean Average Precision (mAP):** Used in the notebook to compare segmentation checkpoint performance.
- **Loss:** Measures the difference between model predictions and ground-truth masks.

## Final Model and Results

The notebook's final model summary reports the following configuration:

| Parameter | Value |
|---|---|
| Architecture | Pretrained Attention R2-UNet |
| Encoder backbone | ResNet34 |
| Batch size | 8 |
| Learning rate | 0.0001 (`1e-4`) |
| Optimizer | AdamW |
| Loss | Combined Dice and BCE |
| Dice loss weight | 0.7 |
| BCE loss weight | 0.3 |

### Reported Performance

| Metric | Result |
|---|---:|
| Pixel Accuracy | 93.53% |
| Mean Dice | 0.8574 |
| Mean IoU | 0.7783 |
| Dataset mAP | 0.9245 |

These are the values recorded in the notebook's final summary. Re-run the evaluation with the required dataset and checkpoints to reproduce them.

## Project Files and Dependencies

The notebook imports project-specific utilities and model implementations, including:

- `util.py` – custom augmentation and dataset-related utilities
- Model definitions for U-Net and attention-based U-Net variants
- Training and evaluation utilities
- Dataset files, JSON annotations, and model checkpoints

The notebook references checkpoint folders under `checkpoints/` and saved metric files such as `mAP_results_augmentation.csv` and `mAP_results_architecture.csv`. These supporting files are not embedded in the notebook and must be supplied separately where required.

## How to Run

1. Clone or download the project repository.
2. Install PyTorch and Torchvision for your system. Choose the installation command appropriate for your operating system and CUDA version from the official PyTorch installation guide.
3. Install the other Python libraries used by the notebook.
4. Place the Nature dataset and its annotations in the expected folder structure.
5. Add the project utility modules and model definitions.
6. Open `Image_Segmentation.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
7. Update dataset and checkpoint paths if necessary.
8. Run the notebook cells in order.

Example installation for the common supporting libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Install PyTorch and Torchvision separately using the official selector: https://pytorch.org/get-started/locally/

> **Important:** The notebook depends on custom modules (such as `util.py`), model definitions, dataset files, and checkpoints. The notebook alone may not be sufficient to run every cell.

## Expected Outputs

- Dataset image and mask visualizations
- Data augmentation comparisons
- Image and mask shape checks
- Model training and evaluation metrics
- Loss, Dice, accuracy, and IoU plots
- mAP comparison plots
- Predicted segmentation masks
- Segmentation evolution visualizations across training epochs

## Conclusion

This project demonstrates a deep-learning approach to segmenting butterflies and squirrels using binary masks. It compares multiple U-Net-based architectures and data augmentation strategies. According to the notebook's final reported results, the **Pretrained Attention R2-UNet with a ResNet34 encoder**, combined Dice and BCE loss, and mask-aware augmentation provides the selected final configuration.

---

*Documentation prepared from the `Image_Segmentation.ipynb` notebook.*
