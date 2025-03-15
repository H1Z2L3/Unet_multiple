# Unet_multiple-
Deep Learning-Based Closed-Loop Surface-Related Multiple Elimination  Method
This project implements a Deep Learning-Based Closed-Loop Surface-Related Multiple Elimination method in Python, which consists of several key components:

ultismultiple.py
This code is responsible for data augmentation. It enhances the original dataset by applying various transformations to create more diverse training examples. This helps improve the generalization ability of the model.

unet3multiple.py
This code defines the network construction. It implements the architecture of the deep learning model used for surface-related multiple estimation, which is based on the U-Net.

trainmultiple.py
This code is used for network training. It trains the model on the prepared data. The training process allows the network to learn how to perform surface-related multiple estimation effectively.

applymultiple.py
This code applies the trained model to new data. It uses the learned weights from the training process to perform surface-related multiple estimation on the test data.

Testing Instructions
To test and run the entire code, follow the steps below:

Train the Model

Run the trainmultiple.py code to start the training process. 
python trainmultiple.py

Apply the Model
After the model is trained, use the applymultiple.py code to apply the trained model to new or test data.
python applymultiple.py
This will output the results of surface-related multiple estimation on the test data, using the trained model.

Then, by subtracting the estimated multiples, precise multiple suppression is achieved.







