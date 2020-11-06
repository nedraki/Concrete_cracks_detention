import tensorflow as tf
import tensorflow.keras
import keras
from PIL import Image, ImageOps
import numpy as np


def classification(img, model):
    # Load the model
    model = keras.models.load_model(model)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    
    data = np.ndarray(shape=(1, 160, 160, 3), dtype=np.float32)
    image = img
    
    ##esizing the image to be at least 160x160 and then cropping from the center
    size = (160, 160)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    data[0] = image_array
    
    ## Note: The image is not normalized because a normalized layer
    ## is included in the uploaded model

    
    # Prediction:
    
    prediction = model.predict(data)
    
    
    # Apply a sigmoid since our model returns logits
    
    prediction = tf.nn.sigmoid(prediction)
    probability = (1- prediction)*100
    prediction = tf.where(probability > 80, 0, 1)
    
    return prediction, probability

#testing
