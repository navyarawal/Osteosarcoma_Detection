# Osteosarcoma Detection with a Convolutional Neural Network

A convolutional neural network that classifies osteosarcoma (bone cancer) histopathology images into three tissue types, built to explore how deep learning can support faster, more consistent pathology screening.

**Authors:** Navya Rawal & Aviva Wang

## The problem

Osteosarcoma is the most common bone cancer in children and young adults. Diagnosis relies on pathologists manually reviewing histology slides — accurate but slow and subjective. This project trains a CNN to classify tissue patches automatically, as a proof of concept for computer-assisted diagnosis.

## The data

Histopathology image tiles labeled into three classes:

- **Non-Tumor**
- **Viable (tumor)**
- **Non-Viable-Tumor**

Images are split into training, validation, and test sets, with CSV files mapping each image to its label.

## Approach

**Preprocessing**
- Images resized to 375×375 and normalized to `[0, 1]`
- Training-time augmentation: ±30° rotation, width/height shifts, horizontal & vertical flips

**Model** (Keras `Sequential`)
- 3 convolutional blocks (32 → 64 → 128 filters), each with max pooling
- Dense layers (512 → 256 units) with 0.7 dropout
- Softmax output over the 3 classes

**Training**
- Adam optimizer, learning rate `1e-5`
- Categorical cross-entropy loss
- `EarlyStopping` + `ReduceLROnPlateau` callbacks

## Results

| Metric | Score |
|---|---|
| Test accuracy | **80.9%** |
| Weighted F1 | **0.81** |
| Test samples | 199 |

A confusion matrix and full classification report are generated in the notebook.

## Tech

`Python` · `TensorFlow / Keras` · `scikit-learn` · `Jupyter`

## Running it

Open `test.ipynb` in Jupyter (or Google Colab) and run the cells top to bottom. You'll need the histopathology dataset in the expected `train/val/test` folder layout with the accompanying label CSVs.

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## Notes & next steps

- Results shown are from a short training run; accuracy should improve with more epochs and a larger model.
- Ideas for future work: transfer learning from a pretrained backbone (e.g. ResNet/EfficientNet), class-imbalance handling, and Grad-CAM to visualize what the network focuses on.
