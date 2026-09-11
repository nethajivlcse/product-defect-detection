# Automated Product Defect Detection

Binary image classification (Normal vs Defective) using transfer learning.

## Problem
A manufacturing company wants to automate defect detection on its production
line. This project classifies product images as **Normal** or **Defective**
using a pre-trained CNN fine-tuned on the provided dataset.

## Approach
- **Preprocessing:** Images resized to 224x224, converted to tensors, and
  normalized using ImageNet mean/std (required since the model backbone is
  ImageNet-pretrained).
- **Split:** Stratified 80/20 train/validation split, so both classes are
  proportionally represented in each set.
- **Augmentation:** Random horizontal/vertical flip, rotation, crop, and
  brightness/contrast jitter — applied to the training set only.
- **Model:** ResNet18 pretrained on ImageNet, used as a frozen feature
  extractor. Only the final fully-connected layer is replaced and trained,
  which keeps training fast and resists overfitting on a small dataset.
- **Training:** Adam optimizer, cross-entropy loss, best-validation-accuracy
  checkpoint kept.
- **Evaluation:** Accuracy, Precision, Recall, F1-score, and a confusion
  matrix on the held-out validation set.
- **Error analysis:** At least 3 misclassified validation images are shown
  with actual vs predicted label and model confidence.

## Project structure
```
studentname_rollno/
├── main.py              # full pipeline: data loading -> training -> evaluation
├── requirements.txt
├── README.md
├── model/
│   └── trained_model.pth
└── results/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── misclassified_examples.png
    └── metrics.txt
```

## How to run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py --data_dir ./dataset --epochs 8
```

`--data_dir` should point to a folder containing one subfolder per class,
e.g. `dataset/Normal/` and `dataset/Defective/`.

Outputs are written automatically to `model/` and `results/`.

## Results
See `results/metrics.txt` for the numeric scores and `results/*.png` for the
training curves, confusion matrix, and misclassified examples.

## Preference
Why PyTorch/torchvision instead of OpenCV: OpenCV is a classical computer vision library built for image processing operations such as edge detection, thresholding, contour analysis, and morphological transformations — it does not provide built-in support for defining, training, or fine-tuning deep neural networks. This project's requirement is a binary classifier built via transfer learning on a pre-trained CNN (Task 2), which requires a deep learning framework capable of loading pretrained weights and updating them through backpropagation. PyTorch, together with torchvision.models, was used because it directly provides ImageNet-pretrained CNN architectures such as ResNet18 that can be loaded, frozen, and fine-tuned in just a few lines of code, making it well suited to the training-time constraints of this task. OpenCV was therefore not used as the classification engine, though it remains a suitable tool for classical, non-learning-based image processing tasks outside this project's scope.

## ResNet18
ResNet18 is an 18-layer Convolutional Neural Network from the ResNet ("Residual Network") family, introduced by Microsoft Research in 2015. It was trained on ImageNet — a dataset of 1.2 million images across 1,000 object categories — and became a widely used backbone for transfer learning because of both its accuracy and its relatively small size.
