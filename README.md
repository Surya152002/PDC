# PDC
# Distributed TensorFlow Model Training using MPI
This repository contains a distributed system implementation of a TensorFlow model training using MPI (Message Passing Interface). The objective of this project is to train a Convolutional Neural Network (CNN) model for image classification using multiple nodes or workers in a distributed environment.

# Requirements
To run this project, you will need the following:

Python 3.x
TensorFlow
mpi4py
OpenCV (cv2)
NumPy
imutils

# Dataset

The code expects the dataset to be organized in separate directories for each class. The images should be in grayscale format and resized to 50x50 pixels. The directory structure should look like the following:


data/
    ├── NORMAL/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── PNEUMONIA/
        ├── image1.png
        ├── image2.png
        └── ...

        
# Running the Code
1. Clone the repository and navigate to the project directory.

2. Ensure you have installed all the required dependencies by running:


pip install -r requirements.txt

3. Adjust the number of workers (nodes) in the code. By default, the code is set up to use 5 workers (4 worker nodes and 1 master node). You can change this according to your distributed environment by modifying the following lines in the code:
python

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

4. Run the code using MPI, for example:

mpiexec -n 5 python distributed_tf_training.py


**Please note that the number of processes (-n) should match the number of worker nodes you want to use.**

# Results

The model training and weight synchronization will be performed in a distributed manner across the specified worker nodes. The master node will receive the updated weights from each worker node and combine them to get the final model weights.

The training progress and validation results will be displayed during the training process.

# Acknowledgments

The model architecture used in this project is a simple Convolutional Neural Network (CNN) with one hidden dense layer.
The code utilizes the mpi4py library for communication between master and worker nodes.

# Disclaimer
This project is for educational purposes and is meant to demonstrate distributed TensorFlow model training using MPI. For production environments or large-scale distributed training, more sophisticated approaches and tools should be used.
