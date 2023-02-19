import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import numpy as np
import cv2
from keras.utils import to_categorical
import telebot
from PIL import Image
import io
import numpy as np

import os
import cv2
from keras.utils import to_categorical

# Load the pre-trained model
model = tf.keras.models.load_model('C:/Users/HP/PycharmProjects/pythonProject3/my_model.h5')

# Create the telegram bot
bot = telebot.TeleBot('6175179776:AAHTDa-s3z4mTL3SHZByXSBErXDtcpbjTUs')

# Handler for the /start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome! Send me a photo of a pokemon and I'll try to recognize it.")

# Handler for image messages
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    # Download the image and convert to numpy array
    file_info = bot.get_file(message.photo[-1].file_id)
    image_file = bot.download_file(file_info.file_path)
    image = Image.open(io.BytesIO(image_file)).convert('RGB')
    image = np.array(image)

    # Resize the image to 224x224 and preprocess for the model
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)

    # Make a prediction using the model
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    print(predicted_class)
    # Send the predicted pokemon back to the user
    label_map = ["bulbasaur", "charmander", "squirtle", "psyduck", "pikachu"]
    pokemon_name = label_map[predicted_class]
    bot.reply_to(message, f"That looks like a {pokemon_name}!")

# Start the bot
bot.polling()