import os

import cv2
import numpy as np
from cv2 import imread
from numpy import random


def loadimgs(path, n=0):
    '''
    path => Path of train directory or test directory
    '''
    x = []
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n

    # we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)

        # every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)

            # read all the images in the current category
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path, cv2.IMREAD_GRAYSCALE)
                category_images.append(image)
                y.append(curr_y)
            try:
                x.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    x = np.stack(x)
    return x, y, lang_dict


def get_batch(batch_size, x, y, categories):
    """Create batch of n pairs, half same class, half different class"""
    n_classes, n_examples, w, h = x.shape

    # randomly sample several classes to use in the batch
    categories = random.choice(n_classes, size=(batch_size,), replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = random.randint(0, n_examples)
        pairs[0][i, :, :, :] = x[category, idx_1].reshape(w, h, 1)
        idx_2 = random.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + random.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = x[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets


def create_batch(batch_size, x, categories):
    pass


if __name__ == '__main__':
    x, y, z = loadimgs("/home/user72/datasets/omniglot-py/images_background/")
    b = get_batch(72, x, y, z)
    # cv2.imshow('image', x[123][0])
    # cv2.waitKey(0)
