# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# import keras
# import matplotlib.pyplot as plt
# import cv2
# import joblib
# import streamlit as st
# import pickle
# from io import StringIO
# train = 'train'
# train_datagen = keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# train_generator = train_datagen.flow_from_directory(
#     train,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical')

# def predict_image(image_path, model_path, class_labels):
#     # model = tf.keras.models.load_model(model_path)
#     # model = joblib.load(model_path)
#     pickle_in = open(model_path, 'rb')
#     model = pickle.load(pickle_in)
#     # Load the original image in RGB format
#     original_image = cv2.imread(image_path)
#     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#     # Resize and normalize the image to match the input shape required by the model
#     resized_image = cv2.resize(original_image, (224, 224))
#     normalized_image = resized_image / 255.0

#     # Reshape the image to match the input shape required by the model
#     reshaped_image = normalized_image.reshape(1, 224, 224, 3)

#     # Make a prediction
#     pred = model.predict(reshaped_image)

#     # Decode the prediction
#     decode = np.argmax(pred)
#     predicted_label = class_labels[decode]

#     # Display the original image, prediction label, and prediction image
#     plt.figure(figsize=(15, 5))

#     # Show the original image
#     plt.subplot(1, 3, 1)
#     plt.imshow(original_image)
   
#     print('Prediction', predicted_label)
#     st.write(predicted_label)
#     plt.axis('off')

#     # Show the grayscale prediction image with prediction label
   

# # Example usage
# class_labels = list(train_generator.class_indices.keys())

# predicted_image = predict_image('snow3.jpeg','big_cats_finalv_.pkl',list(train_generator.class_indices.keys()))




# import mimetypes



# from PIL import Image

# img_file_buffer = st.file_uploader('Upload a PNG image', type=['png','jpg','jpeg','webp'])

# if img_file_buffer is not None:
#     image = Image.open(img_file_buffer)
    
#     img_array = np.array(image)
#     predicted_image = predict_image(img_file_buffer,'big_cats_finalv_.pkl',list(train_generator.class_indices.keys()))
    


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import cv2
import joblib
import streamlit as st
import pickle
from io import BytesIO
from PIL import Image

train = 'train'
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

def predict_image(image, model_path, class_labels):
    # Load the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Check if the image is a file path or BytesIO object
    if isinstance(image, BytesIO):
        image = Image.open(image).convert('RGB')
        image = np.array(image)
    elif isinstance(image, str):
        # Load the original image in RGB format if it's a file path
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to (224, 224)
    resized_image = cv2.resize(image, (224, 224))

    # Normalize the image
    normalized_image = resized_image / 255.0

    # Reshape the image to match the input shape required by the model
    reshaped_image = np.reshape(normalized_image, (1, 224, 224, 3))

    # Make a prediction
    pred = model.predict(reshaped_image)

    # Decode the prediction
    decode = np.argmax(pred)
    predicted_label = class_labels[decode]

    # Display the original image, prediction label, and prediction image
    plt.figure(figsize=(15, 5))

    # Show the original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f'Prediction: {predicted_label}')
    plt.axis('off')

    plt.show()
    st.write('Prediction : ',predicted_label)
    st.image(image, caption=predicted_label)

# Example usage
class_labels = list(train_generator.class_indices.keys())

# Streamlit file uploader and prediction
img_file_buffer = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'webp'])

if img_file_buffer is not None:
    predict_image(img_file_buffer, 'big_cats_finalv_.pkl', class_labels)


