# Chest X-ray AI Classifier

A deep learning tool for classifying chest X-ray images as "Normal" or "Abnormal" using a Convolutional Neural Network (CNN). Includes an interactive notebook for uploading images, running predictions, and visualizing results with confidence scores.

## Features
- Classifies chest X-ray images as Normal or Abnormal
- Uses a simple CNN model (PyTorch)
- Interactive notebook interface for single or batch predictions
- Displays images and prediction confidence
- Easy to use for research, prototyping, or educational purposes

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- numpy
- pillow
- ipywidgets
- Jupyter Notebook

## Installation
Install dependencies using pip:

```bash
pip install torch torchvision numpy pillow ipywidgets
```

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ChestXray-AI-Classifier.git
   cd ChestXray-AI-Classifier/Imaging/notebooks
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `predict-xray-images-pytorch.ipynb` and follow the instructions in the notebook.
4. Upload images or specify a folder to run predictions and view results.

## Model Weights
- Place your trained model weights (e.g., `best_model.pth`) in the notebook directory.
- Update the notebook code if your weights file has a different name.

## Training
To train your own model:
1. Prepare your dataset of chest X-ray images, organized into appropriate folders (e.g., Normal/Abnormal).
2. Create or use a training notebook/script that defines the `SimpleCNN` model and training loop (see the model definition in the prediction notebook).
3. Train the model on your dataset using PyTorch.
4. Save the trained model weights using:
   ```python
   torch.save(model.state_dict(), 'best_model.pth')
   ```
5. Place the saved weights in the notebook directory for inference.

## Learning series 

Part of my learning AI/ML for Medical and Clinical application series.

5. **Binary Classification of X-ray Images (Beginner)**
   - **Description:** Classify chest X-rays as normal or abnormal using basic convolutional neural networks. This project covers image preprocessing, augmentation, and simple model training.
   - **Goals:** Provide a first step toward automated radiology triage.
   - **Methods:** Image normalization, data augmentation, simple CNN architectures.
   - **Potential Impact:** Can help prioritize urgent cases and reduce radiologist workload.
   - **Example ML Techniques:** Basic CNN, transfer learning with pre-trained models.
   - **Key Challenges:** Label noise, small datasets, overfitting.
   - **Evaluation Metrics:** Accuracy, AUROC, confusion matrix.
   - **Possible Extensions:** Multi-class classification, explainable AI overlays, integration with clinical data.
   - **Dataset(s):** *ChestX-ray14, MIMIC-CXR*
   - **Difficulty:** `Beginner`

## License
MIT License 