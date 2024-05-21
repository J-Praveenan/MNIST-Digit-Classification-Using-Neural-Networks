# MNIST-Digit-Classification-Using-Neural-Networks (ANN)

This project aims to classify hand-written digits using an Artificial Neural Network (ANN) implemented with TensorFlow. The MNIST dataset is utilized for training and testing the model.
The project involves building and training a deep learning model to classify hand-written digits (0-9). The model architecture is based on an Artificial Neural Network (ANN).

# Requirements
Python 3.x <br>
TensorFlow
NumPy
Matplotlib
Seaborn
OpenCV
Google Colab (for development and experimentation)
Installation
You can install the required dependencies using pip:

# Dataset
The MNIST dataset is used in this project, which is a widely-used dataset for hand-written digit classification. It consists of 60,000 training images and 10,000 testing images, each of size 28x28 pixels.
The dataset can be loaded using the Keras library

# Training the Model
The model is trained using the training data (X_train and Y_train) obtained from the MNIST dataset. TensorFlow's Keras API is used for building and training the neural network model.

# Evaluation
The trained model is evaluated on the testing data (X_test and Y_test) to measure its performance in classifying hand-written digits.

# Usage
1. Clone the repository:
    [gh repo clone J-Praveenan/MNIST-Digit-Classification-Using-Neural-Networks](https://github.com/J-Praveenan/MNIST-Digit-Classification-Using-Neural-Networks.git)
2. Navigate to the project directory:
    cd Digit-Classification-ANN
3. Install required libraries
    pip install tensorflow numpy matplotlib seaborn opencv-python google-colab
4. Run all the cells
    Results
   
# The performance of the trained model on the testing data is as follows:

Training data accuracy = 98.88 %
Testing data accuracy = 97.54 %
Confusion Matrix:
![ConfusionMetrics](https://github.com/J-Praveenan/MNIST-Digit-Classification-Using-Neural-Networks/assets/138190595/e389b95a-1fdb-4669-b4a2-4efb033d9027)

