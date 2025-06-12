import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files
from google.colab import drive
import json
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d emmarex/plantdisease
!unzip -o plantdisease.zip
data_dir = '/content/plantvillage/PlantVillage'

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_ds = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=64,
    class_mode='categorical',
    seed=42,
    subset='training'
)


validation_ds = tf.keras.utils.image_dataset_from_directory(
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

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Rescaling(1./255, input_shape=(64, 64, 3)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (4,4),activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (1,1), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(512, (2,2),activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512,(2,2), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (2,2), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(512, (4,4), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (4,4), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

optimizer=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy']

              )

epochs = 20
model.fit(train_ds,
    steps_per_epoch=train_ds.samples // 64,
    epochs=epochs,
    validation_data=validation_ds,
    validation_steps=validation_ds.cardinality()
drive.mount('/content/drive')

model.save('/content/drive/MyDrive/plant_disease_model_tf.keras')
with open('/content/drive/MyDrive/plant_disease_model_tf_class_names.json', 'w') as f:
  json.dump(class_names, f)

#-----------------------------------------------

import tensorflow as tf                                                               # Load model from Drive
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from google.colab import files
from google.colab import drive
import json
drive.mount('/content/drive')
model = models.load_model('/content/drive/MyDrive/plant_disease_model_tf.keras')

'''
print('Upload the kaggle.json file\n')
files.upload()
!mkdir -p ~/.kaggle                                                           # Predictions
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d emmarex/plantdisease
!unzip -o plantdisease.zip
data_dir = '/content/plantvillage/PlantVillage'
'''

with open('/content/drive/MyDrive/plant_disease_model_tf_class_names.json', 'r') as f:
    class_names = json.load(f)

def preprocess_img(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(64, 64))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

print('Upload the image for prediction\n')
uploaded=files.upload()
if uploaded:
  image_file_path = list(uploaded.keys())[0]
  processed_image = preprocess_img(image_file_path)
  predictions = model.predict(processed_image)
  predicted_class_index = np.argmax(predictions[0])
  predicted_class_name = class_names[predicted_class_index]
  confidence = predictions[0][predicted_class_index]

  probabilities = predictions[0]
  print("Class Probabilities:")
  for i, prob in enumerate(probabilities):
    print(f"{class_names[i]}: {prob}%")

  print(f'Predicted Label: {predicted_class_name} (Confidence: {confidence}%)')
else:
  print("No image uploaded for prediction.")

