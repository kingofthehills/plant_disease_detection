# Plant Disease Detection

An image classification model that detects diseases in coriander (cilantro) plant leaves using transfer learning with **ResNet101**. Given a leaf image, the model predicts which of several health/disease classes it belongs to.

## Key Features

- Transfer learning on top of **ResNet101** (pretrained on ImageNet), with a custom classification head (`GlobalAveragePooling2D` → `Dense(128, relu)` → `Dense(num_classes, softmax)`, with dropout for regularization).
- **Two-phase training**: the base ResNet101 is first frozen and only the new head is trained, then the last 30 layers of the base model are unfrozen and fine-tuned at a lower learning rate.
- **Data augmentation** via `ImageDataGenerator` (rotation, shifts, shear, zoom, brightness, channel shift, horizontal flip) to improve generalization on a relatively small dataset.
- **Class balancing** using `sklearn.utils.class_weight.compute_class_weight`, so under-represented disease classes are not ignored during training.
- **Training callbacks**: `EarlyStopping` and `ReduceLROnPlateau` to avoid overfitting and adapt the learning rate automatically.
- **Evaluation**: accuracy, precision, recall, F1-score, a full classification report, and a confusion matrix heatmap (via `seaborn`/`matplotlib`).

Based on the dataset split logged during training, the model was trained on **713 images across 6 classes**, validated on **125 images**.

## Tech Stack

- Python
- TensorFlow / Keras (`tensorflow.keras`)
- ResNet101 (`tensorflow.keras.applications`)
- scikit-learn (class weighting, evaluation metrics)
- matplotlib, seaborn (confusion matrix visualization)
- NumPy

## Project Structure

```
plant_disease_detection/
├── plant.py            # Training script: data loading, model build, two-phase training, evaluation
└── coriander.ipynb      # Jupyter/Colab notebook version of the same training pipeline
```

## Setup & Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

A CUDA-capable GPU is recommended for training speed but not required; the script attempts to configure GPU memory limits if a GPU is detected and otherwise runs on CPU.

## Dataset

The scripts expect an image dataset organized into per-class subfolders (compatible with Keras' `ImageDataGenerator.flow_from_directory`), e.g.:

```
coriander dataset/
├── class_1/
├── class_2/
├── ...
```

The dataset path is currently hardcoded in the scripts (`dataset_path = '/workspace/Untitled Folder/coriander datset'`). **Update this path** to point to your own local copy of the coriander leaf image dataset before running. The dataset itself is not included in this repository.

## How to Run

1. Arrange your image dataset into class-labeled subfolders as described above.
2. Edit `dataset_path` in `plant.py` (or the corresponding cell in `coriander.ipynb`) to point to your dataset location.
3. Run the training script:

```bash
python plant.py
```

   or open and run `coriander.ipynb` in Jupyter/Google Colab.

4. The script will:
   - Train the classification head with the ResNet101 base frozen.
   - Fine-tune the last 30 layers of ResNet101 at a lower learning rate.
   - Print validation accuracy, precision, recall, F1-score, and a full classification report.
   - Display a confusion matrix heatmap of the validation predictions.

## Notes

- `img_size` is set to `224x224` and `batch_size` to `32`; these can be adjusted at the top of the script.
- 15% of the dataset is held out for validation (`validation_split=0.15`).
