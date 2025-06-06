This plant disease detector utilizes deep learning and CNNs to detect which disease a plant has. The architecture is as follows : 

1. Pretrained VGG16 architecture with frozen weights, truncated before bottleneck
2. Flatten layer
3. Dropout layer
4. Fully connected layer with ReLU activation
5. Fully connected layer with ReLU activation
6. Fully connected layer with ReLU activation
7. Fully connected layer with Softmax activation

    
Diagram (click to view in detail):     

![plant_disease_model_tf keras](https://github.com/user-attachments/assets/11a6c733-8aa0-4f1d-b20a-e9d6dc74935c)


Two versions of the model are included: one implemented with PyTorch, and one implemented with TensorFlow. The instructions are universal for both versions.

**Note**: The TensorFlow version, which is the version under active development, will save the model at /content/drive/MyDrive/plant_disease_model_tf.keras.     


Instructions: 

1. Paste the code in either detector-pytorch.py or detector-tensorflow.py into a Google Colab notebook.
2. For the TensorFlow version, split the code into two cells: one for training, one for prediction. The segments of the code that are to be split is denoted by a line in the code. The two different segments are specifically designed so that they can be run after disconnecting and reconnecting the runtime, allowing for more versatile usage.
3. Connect to a GPU, such as the T4 (free) or A100 GPU (paid).
4. In kaggle.com, create a new API token if you don't have one already.
5. In Google Colab, upload the kaggle.json file when prompted
6. Enjoy! Please create a pull request if you have any suggestions, or open an issue if you have questions!
