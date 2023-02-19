
Project Report: Pokemon Recognition with Machine Learning deployed on Telegram Bot

Introduction:
The objective of this project is to build a machine learning model that recognizes Pokemon and deploy it as a Telegram bot. The model is trained on a preprocessed dataset containing images of five different Pokemon, namely Bulbasaur, Charmander, Squirtle, Psyduck, and Pikachu. The dataset is split into three subsets for training, validation, and testing. The model is trained on the training data and evaluated on the validation and test data. The trained model is then deployed as a Telegram bot that users can send photos of Pokemon to and receive a response that identifies the Pokemon.

Methodology:
The first step in this project is to preprocess the dataset by resizing the images to a fixed size of 224x224 and save the processed data to separate directories for training, validation, and testing. The preprocessed data is saved as numpy arrays using the numpy library.

Next, the preprocessed data is loaded from the numpy arrays and split into training and validation sets. The training data is used to train a convolutional neural network (CNN) model using the tensorflow.keras library. The CNN model consists of two convolutional layers, two max-pooling layers, and two dense layers.

The trained model is evaluated on the validation set to determine its performance. The performance is measured using accuracy as the evaluation metric. The best model is saved to a file for later use.

The final step in this project is to deploy the trained model as a Telegram bot using the telebot library. The bot is created and set up to handle two types of messages: /start and image messages. When the user sends the /start message, the bot responds with a welcome message. When the user sends an image message, the bot downloads the image and processes it using the trained model. The predicted Pokemon is sent back to the user as a response.

Results:
The CNN model achieved an accuracy of 96.77% on the validation set and 94.40% on the test set. The model is able to recognize Bulbasaur, Charmander, Squirtle, Psyduck, and Pikachu with high accuracy. The model is deployed as a Telegram bot and can recognize Pokemon in images sent by users.

Conclusion:
The machine learning model for Pokemon recognition and its deployment as a Telegram bot was successful. The model achieved high accuracy on the validation and test sets and was able to recognize Pokemon in images sent by users. The Telegram bot deployment provides a user-friendly interface for users to interact with the model and obtain Pokemon recognition results.
As the next step for the project development i plan to include more pokemons for recognition
