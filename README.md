This plant disease detector utilizes deep learning and CNNs to detect which disease a plant has. The architecture is as follows : 

1. VGG16 base model, with frozen weights (from ImageNet)
2. Average pooling layer
3. Batch normalization layer
4. Dropout layer
5. Fully connected layer (ReLU activation)
6. Dropout layer
7. Fully connected layer (ReLU activation)
8. Dropout layer
9. Fully connected layer (ReLU activation)
10. Output layer

    
Diagram:     

![Model Architecture](https://github.com/user-attachments/assets/c9c69f75-8415-4a9a-a41e-a25f489d7c5b)

Instructions: 

1. Go to https://www.kaggle.com/code/lithio67/plant-disease-classifier to view the Kaggle version.
2. Select a GPU
3. Enjoy!
