import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import numpy as np

import os
import cv2
from keras.utils import to_categorical

input_dir = "C:/Users/HP/Desktop/pokemons/unprocessed"
output_dir = "C:/Users/HP/Desktop/pokemons/preprocessed"

for split in ["train", "valid"]:
    split_dir = os.path.join(input_dir, split)
    output_split_dir = os.path.join(output_dir, split)
    if not os.path.exists(output_split_dir):
        os.makedirs(output_split_dir)
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        output_class_dir = os.path.join(output_split_dir, class_name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for filename in os.listdir(class_dir):
            input_path = os.path.join(class_dir, filename)
            output_path = os.path.join(output_class_dir, filename)
            image = cv2.imread(input_path)
            image = cv2.resize(image, (224, 224))
            cv2.imwrite(output_path, image)
    import numpy as np

    train_dir = "C:/Users/HP/Desktop/pokemons/preprocessed/train"
    valid_dir = "C:/Users/HP/Desktop/pokemons/preprocessed/valid"

    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []

    # Loop through the train directory and load the images and labels
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        class_index = str(class_name)
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            image = cv2.imread(image_path)
            train_data.append(image)
            train_labels.append(class_index)

    # Loop through the validation directory and load the images and labels
    for class_name in os.listdir(valid_dir):
        class_dir = os.path.join(valid_dir, class_name)
        class_index = str(class_name)
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            image = cv2.imread(image_path)
            valid_data.append(image)
            valid_labels.append(class_index)

    # Convert the lists to numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    valid_data = np.array(valid_data)
    valid_labels = np.array(valid_labels)

    # Save the data to npy files
    np.save("train_data.npy", train_data)
    np.save("train_labels.npy", train_labels)
    np.save("valid_data.npy", valid_data)
    np.save("valid_labels.npy", valid_labels)


# Load the preprocessed data
train_data = np.load('C:/Users/HP/Desktop/pokemons/preprocessed/train_data.npy')
train_labels = np.load('C:/Users/HP/Desktop/pokemons/preprocessed/train_labels.npy')
val_data = np.load('C:/Users/HP/Desktop/pokemons/preprocessed/valid_data.npy')
val_labels = np.load('C:/Users/HP/Desktop/pokemons/preprocessed/valid_labels.npy')

label_map = {'bulbasaur': 0, 'charmander': 1, 'squirtle': 2, 'psyduck': 3, 'pikachu': 4}

train_labels = [label_map[label] for label in train_labels]
val_labels = [label_map[label] for label in val_labels]

train_labels = to_categorical(train_labels, num_classes=5)
val_labels = to_categorical(val_labels, num_classes=5)
# Define the model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=50, validation_data=(val_data, val_labels))
test_dir = "C:/Users/HP/Desktop/pokemons/preprocessed/test"


test_data = []
test_labels = []

# Loop through the test directory and load the images and labels
for class_name in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, class_name)
    class_index = str(class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path)
        test_data.append(image)
        test_labels.append(class_index)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

np.save("test_data.npy", test_data)
np.save("test_labels.npy", test_labels)

# Convert target to a binary class matrix
test_labels = [label_map[label] for label in test_labels]
binary_test_labels = to_categorical(test_labels)
np.save("C:/Users/HP/Desktop/pokemons/preprocessed/binary_test_labels.npy", binary_test_labels)
# Evaluate the model
test_data = np.load('C:/Users/HP/Desktop/pokemons/preprocessed/test_data.npy')
binary_test_labels = np.load('C:/Users/HP/Desktop/pokemons/preprocessed/binary_test_labels.npy')

test_loss, test_acc = model.evaluate(test_data, binary_test_labels)

# Save the model
model.save('my_model.h5')
