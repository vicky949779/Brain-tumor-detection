# # import cv2
# # import tensorflow as tf
# # from tensorflow import keras
# # from PIL import Image
# # from keras.models import load_model
# # from PIL import Image
# # import numpy as np
# # from keras.models import Sequential


# # model=load_model('BrainTumor10EpochsCategorical.h5')
# # # model = load_model('C:\\Users\\vigneshkumar\\Downloads\\BTD code - Copy\\BrainTumor10EpochsCategorical.h5')

# # # image=cv2.imread('C:\\Users\\SOWMIYA. P K\\Downloads\\brain tumor images\\pred\\pred45.jpg')
# # image=cv2.imread('C:\\Users\\vigneshkumar\\Downloads\\BTD code - Copy\\datasets\\pred\\pred45.jpg')

# # img=Image.fromarray(image)
# # img=img.resize((64,64))
# # img=np.array(img)

# # # print(img)

# # # model=Sequential()
# # input_img=np.expand_dims(img, axis=0)
# # predict_img=model.predict(input_img)
# # classes_img=np.argmax(predict_img,axis=1)

# # if classes_img==[1]:
# #     print("Brain Tumor")
# # else:
# #     print("Not a Brain Tumor")




# # import cv2
# # import tensorflow as tf
# # from tensorflow import keras
# # from PIL import Image
# # from keras.models import load_model
# # import numpy as np

# # # Load the pre-trained model
# # model = load_model('BrainTumor10EpochsCategorical.h5')

# # # Load and preprocess the input image
# # image = cv2.imread('C:\\Users\\vigneshkumar\\Downloads\\BTD code - Copy\\datasets\\pred\\pred45.jpg')
# # img = Image.fromarray(image)
# # img = img.resize((64, 64))
# # img = np.array(img) / 255.0  # Normalize pixel values

# # # Expand dimensions to match the model's expected input shape
# # input_img = np.expand_dims(img, axis=0)

# # # Predict the class probabilities
# # predict_img = model.predict(input_img)

# # # Get the predicted class
# # predicted_class = np.argmax(predict_img, axis=1)[0]

# # # Print the prediction probability for the "Brain Tumor" class
# # print("Prediction Probability for Brain Tumor:", predict_img[0][predicted_class])

# # # Check if the predicted class indicates a brain tumor
# # if predicted_class == 1:
# #     print("Brain Tumor")
# # else:
# #     print("Not a Brain Tumor")

# # -------code cahnges-----------------------

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


# Load the pre-trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Load and preprocess the input image
# image = cv2.imread('C:\\Users\\vigneshkumar\\Downloads\\BTD code - Copy\\datasets\\pred\\pred45.jpg')
image = cv2.imread('C:\\Users\\vigneshkumar\\OneDrive\\Desktop\\Mini project\\Source Code\\BTD code\\datasets\\no\\001.png')
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img) / 255.0  # Normalize pixel values

# Expand dimensions to match the model's expected input shape
input_img = np.expand_dims(img, axis=0)

# Predict the class probabilities
predict_img = model.predict(input_img)

# Get the predicted class
predicted_class = np.argmax(predict_img, axis=1)[0]

# Print the prediction probability for the "Brain Tumor" class
print("Prediction Probability for Brain Tumor:", predict_img[0][predicted_class])

# Check if the predicted class indicates a brain tumor
if predicted_class == 1:
    print("Brain Tumor")
else:
    print("Not a Brain Tumor")
