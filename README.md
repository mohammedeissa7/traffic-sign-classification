# Traffic Sign Recognition System

## Overview
This project presents an Image Classification System designed to recognize and classify traffic signs in real-time. It leverages a Convolutional Neural Network (CNN) model to accurately identify different traffic sign categories, helping in applications such as autonomous driving and traffic monitoring.

## Features
- **Real-time Classification**: The system can classify traffic signs from live camera feed or static images.
- **High Accuracy**: Utilizes a trained CNN model for precise recognition of 43 different traffic sign categories.
- **Preprocessing Techniques**: Includes grayscale conversion, histogram equalization, and normalization to enhance image quality.
- **Data Augmentation**: Uses image transformations to improve model robustness.
## Project Structure
├── myData/                  # Training images (organized by class)  
├── labels.csv               # Class IDs and sign names  
├── TRAIN.ipynb              # Model training notebook  
├── MAIN.ipynb              # Real-time classification notebook  
├── model_trained.p          # Trained model (pickle)  
└── requirements.txt         # Dependencies

## Dataset
The model is trained on a dataset containing 43 different classes of traffic signs. Each class includes multiple images of the respective sign under various conditions. The dataset is split into training, validation, and test sets to ensure robust model performance.

## Model Architecture
The CNN model consists of:
- **Convolutional Layers**: Four layers with ReLU activation for feature extraction.
- **Max Pooling Layers**: For dimensionality reduction.
- **Dropout Layers**: To prevent overfitting.
- **Fully Connected Layers**: For final classification.

## Requirements
To run this project, you need the following libraries:
- OpenCV (`cv2`)
- NumPy (`numpy`)
- TensorFlow (`tensorflow`)
- scikit-learn (`sklearn`)
- Matplotlib (`matplotlib`)
- Pandas (`pandas`)
- Pickle (`pickle`)

Install the dependencies using:
```bash
pip install opencv-python numpy tensorflow scikit-learn matplotlib pandas
```

## Usage
1. **Training the Model**:
    - Run the `TRAIN.ipynb` notebook to train the CNN model on the dataset.
    - The trained model will be saved as `model_trained.p`.

2. **Real-time Classification**:
    - Run the `MAIN.ipynb` notebook to start the live traffic sign recognition using your webcam.
    - The system will display the classified sign name and confidence level in real-time.

## Results
- The model achieves high accuracy on the test set, as shown in the training history plots.
- Real-time classification is demonstrated with a confidence threshold to ensure reliable predictions.

## Useful Links
- [Original Paper](https://www.cell.com/heliyon/fulltext/S2405-8440(22)03080-8?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2405844022030808%3Fshowall%3Dtrue)
- [Dataset](https://drive.google.com/file/d/1AZeKw90Cb6GgamTBO3mvDdz6PjBwqCCt/view)
