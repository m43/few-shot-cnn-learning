# -*- coding: utf-8 -*-

import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision import datasets
from tqdm.auto import tqdm


class OmniglotDataLoaderCreator:
    """
    Class that takes care of creating DataLoaders for Omniglot dataset
    """

    BACKGROUND_FOLDER_NAME = "omniglot-py/images_background"
    EVALUATION_FOLDER_NAME = "omniglot-py/images_evaluation"

    def __init__(self, data_dir, train_samples=3000, validation_samples=1000, validation_shots=320,
                 test_shots=400, train_affine_distortions=False, train_affine_distortions_number=8):
        self.data_dir = data_dir

        self.train_samples = train_samples
        self.validation_samples = validation_samples
        self.validation_shots = validation_shots
        self.test_shots = test_shots

        self.train_affine_distortions = train_affine_distortions
        self.train_affine_distortions_number = train_affine_distortions_number

        datasets.Omniglot(root=self.data_dir, download=True, background=True)
        background_location = os.path.join(self.data_dir, self.BACKGROUND_FOLDER_NAME)
        print("Preprocessing background alphabet", flush=True)
        self._train_alphabets, self._train_alphabet_dict = self._load_alphabets_from_disk(background_location)

        datasets.Omniglot(root=self.data_dir, download=True, background=False)
        print("Preprocessing evaluation alphabet", flush=True)
        evaluation_location = os.path.join(self.data_dir, self.EVALUATION_FOLDER_NAME)
        self._eval_alphabets, self._eval_alphabet_dict = self._load_alphabets_from_disk(evaluation_location)

    def load_train(self, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        print("Loading train loader", flush=True)
        return DataLoader(self._load_train_dataset(), batch_size, shuffle)

    def load_test(self, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        print("Loading oneshot test loader", flush=True)
        return DataLoader(self._load_test_dataset(), batch_size, shuffle)

    def load_validation(self, oneshot: bool, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
        # alphabets 31-40
        # writers 13-16
        # uniform alphabet distribution
        # uniform label (0 or 1) distribution

        alphabets = self._eval_alphabets[:10]  # Take only first 10 for validation, rest is for test
        writers = list(range(13, 16 + 1))

        if oneshot:
            print("Loading oneshot validation loader", flush=True)
            val_dataset = self._load_one_shot_dataset(self._eval_alphabet_dict, alphabets, self.validation_shots,
                                                      20, writers)
            return DataLoader(val_dataset, batch_size, shuffle)

        else:
            print("Loading validation loader", flush=True)
            val_dataset = self._load_uniform_dataset_from_alphabets(self._eval_alphabet_dict, alphabets,
                                                                    self.validation_samples, writers)
            return DataLoader(val_dataset, batch_size, shuffle)

    def _load_test_dataset(self):
        # alphabets 41-50
        # writers 17-20
        # uniform alphabet distribution

        alphabets = self._eval_alphabets[10:]  # Take only the last 10 alphabets for test
        writers = list(range(17, 20 + 1))  # Take only the last 4 writers

        # TODO should I add batch here
        return self._load_one_shot_dataset(self._eval_alphabet_dict, alphabets, self.test_shots, 20, writers)

    def _load_train_dataset(self):
        # alphabets 1-30
        # writers 1-12
        # uniform alphabet distribution
        # uniform label (0 or 1) distribution

        alphabets = self._train_alphabets
        writers = list(range(1, 12 + 1))

        dataset = self._load_uniform_dataset_from_alphabets(self._train_alphabet_dict, alphabets, self.train_samples,
                                                            writers)

        if self.train_affine_distortions:
            # TODO
            raise Exception("Not implemented yet")

        return dataset

    @staticmethod
    def _load_alphabets_from_disk(path: str) -> (Dict, List):
        alphabet_dict = {}
        alphabets = sorted(os.listdir(path))

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0,), (1,))  # TODO or try with whole dataset
        ])

        for alphabet in tqdm(alphabets):
            alphabet_location = os.path.join(path, alphabet)
            for char_idx, char in enumerate(sorted(os.listdir(alphabet_location))):
                char_location = os.path.join(alphabet_location, char)
                for writer_idx, writer in enumerate(sorted(os.listdir(char_location))):
                    image_location = os.path.join(char_location, writer)
                    if alphabet not in alphabet_dict:
                        alphabet_dict[alphabet] = {}
                    if char_idx not in alphabet_dict[alphabet]:
                        alphabet_dict[alphabet][char_idx] = {}
                    alphabet_dict[alphabet][char_idx][writer_idx] = np.asarray(transforms(Image.open(image_location)))
        return alphabets, alphabet_dict

    @staticmethod
    def _load_one_shot_dataset(alphabet_dict: Dict, alphabets: List[str], shots: int, n_way: int,
                               writer_range: List[int]):

        data = []
        for _ in tqdm(range(shots // n_way)):
            alphabet = random.choice(alphabets)
            characters = random.sample(list(range(len(alphabet_dict[alphabet]))), n_way)
            drawer_1_idx, drawer_2_idx = random.sample(writer_range, 2)
            drawer_1_idx, drawer_2_idx = drawer_1_idx - 1, drawer_2_idx - 1

            data.append([[], []])
            for i in range(n_way):
                data[-1][0].append(alphabet_dict[alphabet][characters[i]][drawer_1_idx])
                data[-1][1].append(alphabet_dict[alphabet][characters[i]][drawer_2_idx])

        return TensorDataset(torch.tensor(data))

    @staticmethod
    def _load_uniform_dataset_from_alphabets(alphabet_dict: Dict, alphabets: List[str], samples: int,
                                             writer_range: List[int]):
        data = []
        labels = []
        for i in tqdm(range(samples // 2 // len(alphabets))):
            for alphabet in alphabets:  # uniform number of training examples per alphabet

                char_1_idx = random.randint(1, len(alphabet_dict[alphabet])) - 1
                char_2_idx = random.randint(1, len(alphabet_dict[alphabet]) - 1) - 1
                if char_1_idx == char_2_idx:  # make sure that char1 is not the same as char2
                    char_2_idx = len(alphabet_dict[alphabet]) - 1

                writer_1_idx = random.choice(writer_range) - 1
                writer_2_idx = random.choice(writer_range) - 1  # can be the same as writer1

                data.append([alphabet_dict[alphabet][char_1_idx][writer_1_idx],
                             alphabet_dict[alphabet][char_1_idx][writer_2_idx]])
                labels.append(0)

                data.append([alphabet_dict[alphabet][char_1_idx][writer_1_idx],
                             alphabet_dict[alphabet][char_2_idx][writer_2_idx]])
                labels.append(1)

        return TensorDataset(torch.tensor(data), torch.tensor(labels))


class Visualizer:

    @staticmethod
    def visualize_next_batch(loader):
        img_pairs, labels = next(iter(loader))
        grid = torchvision.utils.make_grid([torch.cat([pair[0], pair[1]], dim=2) for pair in img_pairs], nrow=5)
        plt.figure(figsize=(18, 180))
        plt.axis('off')
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()

    @staticmethod
    def visualize_next_oneshot_batch(oneshot_loader):
        oneshot_pairs = next(iter(oneshot_loader))[0]
        pair = oneshot_pairs[0]
        grid = torchvision.utils.make_grid([torch.cat(
            [torchvision.utils.make_grid(pair[0], nrow=5), torchvision.utils.make_grid(pair[1], nrow=5)], dim=2) for
            pair in oneshot_pairs], nrow=1)
        plt.figure(figsize=(18, 180))
        plt.axis('off')
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()


if __name__ == '__main__':
    torch.manual_seed(72)

    print("Testing omniglot dataloader")
    omniglot_dataloader_creator = OmniglotDataLoaderCreator("./data/", 30000, 10000, 320, 400, False, 8)
    print("Preprocessing done.")
    print()

    print("Creating dataloaders:")
    train_loader = omniglot_dataloader_creator.load_train(100, False)
    val_loader = omniglot_dataloader_creator.load_validation(False, 100, False)
    val_oneshot_loader = omniglot_dataloader_creator.load_validation(True, 5, False)
    test_oneshot_loader = omniglot_dataloader_creator.load_test(5, False)

    list(iter(train_loader.dataset))
    Visualizer.visualize_next_batch(train_loader)

    list(iter(val_loader))
    list(iter(val_oneshot_loader))
    Visualizer.visualize_next_batch(val_loader)
    Visualizer.visualize_next_oneshot_batch(val_oneshot_loader)

    list(iter(test_oneshot_loader))
    Visualizer.visualize_next_oneshot_batch(test_oneshot_loader)

    ## PLAYGROUND
    data = next(iter(test_oneshot_loader))
    new_shape = (30, 1, 105, 105)
    ae = data[0][0][0].expand(new_shape)
    grid = torchvision.utils.make_grid(ae, nrow=5)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
    # perfect!
