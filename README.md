# Pneumonia-Detection-from-Chest-X-Ray-Images-By-CNN
Objective: 
Develop a deep learning model using Convolutional Neural Networks (CNN) to classify chest X-ray 
images as either “Pneumonia” or “Normal”. This case study aims to introduce students to the 
application of CNN in medical image analysis. 
Learning Outcomes: 
Understand the use of CNNs in image classification. 
Practice preprocessing and data augmentation techniques for medical imaging. 
Evaluate a model using accuracy, confusion matrix, and classification report. 
Interpret model outputs and understand model performance on imbalanced classes. 
Dataset Information: 
Name: Chest X-Ray Images (Pneumonia) 
Source: Kaggle (by Paul Timothy Mooney) 
Data Format: JPEG images classified into two folders — Pneumonia and Normal. 
Subfolders: 
train/ 
val/ 
test/ 
Assignment Tasks: 
Download and explore the dataset. Examine the distribution of the Normal and Pneumonia classes in 
each subset (train/val/test). 
Preprocess the data: 
Resize images (e.g., to 150x150 or 224x224 pixels). 
Apply normalization and data augmentation (rotation, flipping, etc.). 
Build a CNN using Keras or PyTorch: 
Suggested layers: Conv2D, MaxPooling2D, Flatten, Dense, Dropout. 
Use ReLU activation and Softmax/Sigmoid as needed. 
Train the model: 
Use binary cross-entropy loss and appropriate optimizer (e.g., Adam). 
Track training and validation accuracy and loss 
Evaluate the model: 
Use the test set to evaluate performance. 
Show confusion matrix, precision, recall, and F1-score. 
Interpret Results: 
Discuss overfitting/underfitting. 
Reflect on class imbalance and performance. 
Optional Extension: 
Use Transfer Learning with pretrained models like VGG16, ResNet50, or MobileNet. 
Implement Grad-CAM or saliency maps to visualize how the model makes decisions.
