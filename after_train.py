# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 00:44:16 2023

@author: deshm
"""

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("C:/Users/deshm/brain_tumor_detection_cnn.h5")

# Define the category labels
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Perform predictions on new data
# Assuming you have a new image you want to predict
new_image_path = ("C:/Users/deshm/.spyder-py3/Training/meningioma/Tr-me_0031.jpg")

new_image = cv2.imread(new_image_path)
resized_image = cv2.resize(new_image, (150, 150))  # Resize to (150, 150)
input_image = np.expand_dims(resized_image, axis=0)  # Reshape to (1, 150, 150, 3)

# Make predictions on the new image
predictions = model.predict(input_image)
predicted_category = np.argmax(predictions)

# Assuming you have the actual category for the new image
actual_category = 1  # Replace with the actual category index

# Print the predicted and actual categories
print("Predicted category:", categories[predicted_category])
print("Actual category:", categories[actual_category])

# Compare the predicted and actual categories
if predicted_category == actual_category:
    print("Prediction is correct!")
else:
    print("Prediction is incorrect!")
