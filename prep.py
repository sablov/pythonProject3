import os
import cv2

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

# Convert test labels to a binary class matrix
binary_test_labels = to_categorical(test_labels)

# Save the binary_test_labels to file
np.save("C:/Users/HP/Desktop/pokemons/preprocessed/binary_test_labels.npy", binary_test_labels)