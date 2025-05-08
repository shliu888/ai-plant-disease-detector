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

def Model(num_classes):
    model = models.Sequential()
    model.add(layers.Rescaling(1./255, input_shape=(64, 64, 3))) 
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))  
    model.add(layers.MaxPooling2D((2, 2)))  
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same')) 
    model.add(layers.MaxPooling2D((2, 2)))  
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))  
    model.add(layers.Dense(512, activation='relu'))  
    model.add(layers.Dense(128, activation='relu'))  
    model.add(layers.Dense(64, activation='relu'))  
    model.add(layers.Dense(num_classes, activation='softmax'))  
    return model

model = Model(num_classes)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 20
model.fit(train_ds,validation_data=val_ds, epochs=epochs)

model.save('/content/plant_disease_model_tf.h5')


def preprocess_img(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(64, 64))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

if __name__ == '__main__':
  
    model = tf.keras.models.load_model('/content/plant_disease_model_tf.h5')

    
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
