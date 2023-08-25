import os
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array, image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGES_DIR = Path("dataset/oxford-iiit-pet/images")
LABELS_DIR = Path("dataset/oxford-iiit-pet/annotations/trimaps")


def display_target(target_array):
    # original labels are 1 2 3. subtract 1
    # multiply by 127 to make it ranges from 0 to 254
    normalized_array = (target_array.astype("uint8") - 1) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])
    plt.show()


def path_to_input_image(path, img_size):
    return img_to_array(load_img(path, target_size=img_size))


def path_to_target(path, img_size):
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1  # subtract 1 so labels start from 0
    return img


input_img_paths = sorted(
    [
        os.path.join(IMAGES_DIR, fname)
        for fname in os.listdir(IMAGES_DIR)
        if fname.endswith(".jpg")
    ]
)


target_paths = sorted(
    [
        os.path.join(LABELS_DIR, fname)
        for fname in os.listdir(LABELS_DIR)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

# print(len(input_img_paths), len(target_paths))

# print(img_to_array(load_img(input_img_paths[9], color_mode="grayscale")).shape)
# print(img_to_array(load_img(target_paths[9], color_mode="grayscale")).shape)

# display image
# plt.axis("off")
# plt.imshow(load_img(input_img_paths[9]))
# plt.show()

# display label
# labels = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
# display_target(labels)

img_size = (200, 200)
num_imgs = len(input_img_paths)

# shuffle img and labels with the same order
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

# load image to float32 array with 3 channels
# load target to uint8 array with 1 channel
input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i], img_size)
    targets[i] = path_to_target(target_paths[i], img_size)

# print(input_imgs.shape)
# print(targets.shape)

