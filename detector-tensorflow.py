import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from google.colab import files
from google.colab import drive
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d emmarex/plantdisease
!unzip -o plantdisease.zip
data_dir = '/content/plantvillage/PlantVillage'

train_ds = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(64, 64),
    batch_size=64,
    shuffle=True,
    validation_split=0.2,
    subset='training',
    seed=42
)

val_ds = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(64, 64),
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
    validation_split=0.2,
    subset='validation',
    seed=42
)

class_names = train_ds.class_names
num_classes = len(class_names)


model = models.Sequential()
model.add(layers.Rescaling(1./255, input_shape=(64, 64, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))




model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 20
model.fit(train_ds,validation_data=val_ds, epochs=epochs)
drive.mount('/content/drive')

model.save('/content/drive/MyDrive/plant_disease_model_tf.keras')

if 'history' in locals(): 
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
else:
    print("Model was not trained in this session.  No training history to plot.")

#------------------------------------------------------------------------------------------------

import tensorflow as tf                                                               # Load model from Drive
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from google.colab import files
from google.colab import drive
drive.mount('/content/drive')
model = tf.keras.models.load_model('/content/drive/MyDrive/plant_disease_model_tf.keras')  


import matplotlib.pyplot as plt
files.upload()
!mkdir -p ~/.kaggle                                                           # Predictions
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d emmarex/plantdisease
!unzip -o plantdisease.zip
data_dir = '/content/plantvillage/PlantVillage'

train_ds = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(64, 64),
    batch_size=64,
    shuffle=True,
    validation_split=0.2,
    subset='training',
    seed=42
)

val_ds = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(64, 64),
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
    validation_split=0.2,
    subset='validation',
    seed=42
)

class_names = train_ds.class_names
num_classes = len(class_names)

def preprocess_img(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(64, 64))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

model = tf.keras.models.load_model('/content/drive/MyDrive/plant_disease_model_tf.keras')
from google.colab import files
uploaded = files.upload()

if uploaded:
  image_file_path = list(uploaded.keys())[0]
  processed_image = preprocess_img(image_file_path)
  predictions = model.predict(processed_image)
  predicted_class_index = np.argmax(predictions[0])
  predicted_class_name = class_names[predicted_class_index]
  confidence = predictions[0][predicted_class_index]

  print(f'Predicted Label: {predicted_class_name} (Confidence: {confidence}%)')
else:
  print("No image uploaded for prediction.")

