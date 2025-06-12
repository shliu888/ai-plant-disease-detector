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

F1 Score: 98.33%         

Training time: 1 hour
    
Diagram:     

![Model Architecture](https://github.com/user-attachments/assets/c9c69f75-8415-4a9a-a41e-a25f489d7c5b)

Validation set accuracy by class:
<img width="895" alt="Screenshot 2025-06-12 at 3 08 01 PM" src="https://github.com/user-attachments/assets/a63b3194-0114-4f3b-9959-e853742e8095" />    


The notebook is also available at [https://www.kaggle.com/code/lithio67/plant-disease-classifier](url).

Feel free to create a pull request with any suggestions or open an issue.
