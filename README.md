This plant disease detector utilizes deep learning and CNNs to detect which disease a plant has. The architecture is as follows : 

  1. Convolution layer 
  2. Max pool layer 
  3. Convolution layer 
  4. Max pool layer  
  5. Dropout layer 
  6. Fully connected layer with ReLU activation
  7. Fully connected layer with ReLU activation
  8. Fully connected layer with ReLU activation
  9. Fully connected layer with Softmax activation

Two versions of the model are included: one implemented with PyTorch, and one implemented with TensorFlow. The instructions are universal for both versions.

**Note**: The TensorFlow version, which is the version under active development, will save the model at /content/drive/MyDrive/plant_disease_model_tf.keras.
Instructions: 

1. Paste the code in either detector-pytorch.py or detector-tensorflow.py into a Google Colab notebook.
2. For the TensorFlow version, split the code into two cells: one for training, one for prediction. The segments of the code that are to be split is denoted by a line.
3. Connect to a GPU, such as the T4 (free) or A100 GPU (paid).
4. In kaggle.com, create a new API token if you don't have one already.
5. In Google Colab, upload the kaggle.json file when prompted
6. Enjoy! Please create a pull request if you have any suggestions, or open an issue if you have questions!
