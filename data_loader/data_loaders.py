# -*- coding: utf-8 -*-
import os
import random
from datetime import datetime
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

    BACKGROUND_FOLDER_NAME = "omniglot-py/images_background"
    EVALUATION_FOLDER_NAME = "omniglot-py/images_evaluation"

    TRANSFORMATIONS = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.9217844519089332,), (0.26851047769786074,), inplace=True)
    ])

    def load_train(self, batch_size: int = 1, shuffle: bool = False, num_workers: int = 6) -> DataLoader:
        print("Loading train loader", flush=True)
        return DataLoader(self._load_train_dataset(), batch_size, shuffle, num_workers=num_workers)

    def _load_train_oneshot(self, batch_size: int = 1, shuffle: bool = False, num_workers: int = 6):
        # alphabets 01-30
        # writers 01-12
        # uniform alphabet distribution

        alphabets = self._train_alphabets
        writers = list(range(1, 12 + 1))

        dataset = self._load_one_shot_dataset(self._train_alphabet_dict, alphabets, self.test_shots, 20, writers)
        return DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)

    def _load_train_dataset(self):
        # alphabets 1-30
        # writers 1-12
        # uniform alphabet distribution
        # uniform label (0 or 1) distribution

        alphabets = self._train_alphabets
        writers = list(range(1, 12 + 1))

        dataset = self._load_uniform_dataset_from_alphabets(self._train_alphabet_dict, alphabets, self.train_samples,
                                                            writers, self.train_affine_distortions,
                                                            self.train_affine_distortions_number)

        return dataset

    @staticmethod
    def _generate_random_affine_transform():
        degrees = 10
        translate = (0.1, 0.1)  # TODO what means (-2,2) in original paper?
        scale = (0.8, 1.2)
        shear = (-0.3, 0.3)

        if random.getrandbits(1):
            degrees = 0
        if random.getrandbits(1):
            translate = None
        if random.getrandbits(1):
            scale = None
        if random.getrandbits(1):
            shear = None

        return torchvision.transforms.Compose(
            [torchvision.transforms.RandomAffine(degrees, translate, scale, shear, fillcolor=255)])

    @staticmethod
    def _load_uniform_dataset_from_alphabets(alphabet_dict: Dict, alphabets: List[str], samples: int,
                                             writer_range: List[int], train_affine_distortions: bool = False,
                                             train_affine_distortions_number=8):
        data = []
        labels = []
        for _ in tqdm(range(samples // 2 // len(alphabets))):
            for alphabet in alphabets:  # uniform number of training examples per alphabet

                char_1_idx = random.randint(1, len(alphabet_dict[alphabet])) - 1
                char_2_idx = random.randint(1, len(alphabet_dict[alphabet]) - 1) - 1
                if char_1_idx == char_2_idx:  # make sure that char1 is not the same as char2
                    char_2_idx = len(alphabet_dict[alphabet]) - 1

                drawer_1_idx, drawer_2_idx = random.sample(writer_range, 2)
                drawer_1_idx, drawer_2_idx = drawer_1_idx - 1, drawer_2_idx - 1
                # writer_1_idx = random.choice(writer_range) - 1
                # writer_2_idx = random.choice(writer_range) - 1  # can be the same as writer1

                data.append([alphabet_dict[alphabet][char_1_idx][drawer_1_idx],
                             alphabet_dict[alphabet][char_1_idx][drawer_2_idx]])
                labels.append(1.)

                data.append([alphabet_dict[alphabet][char_1_idx][drawer_1_idx],
                             alphabet_dict[alphabet][char_2_idx][drawer_2_idx]])
                labels.append(0.)

        if train_affine_distortions:
            new_data = []
            new_labels = []
            for i in tqdm(range(len(data))):
                pair = data[i]
                label = labels[i]
                # new_data.append(pair)
                # new_labels.append(label)
                for _ in range(train_affine_distortions_number):
                    t1 = OmniglotDataLoaderCreator._generate_random_affine_transform()
                    t2 = OmniglotDataLoaderCreator._generate_random_affine_transform()
                    # new_data.append([t1(pair[0]), t2(pair[1])])
                    # new_labels.append(label)
                    data.append([t1(pair[0]), t2(pair[1])])
                    labels.append(label)
            # data = new_data
            # labels = new_labels

        for i in range(len(data)):
            first = OmniglotDataLoaderCreator.TRANSFORMATIONS(data[i][0])
            second = OmniglotDataLoaderCreator.TRANSFORMATIONS(data[i][1])
            data[i] = torch.cat([first.unsqueeze(0), second.unsqueeze(0)]).unsqueeze(0)

        return TensorDataset(torch.cat(data), torch.tensor(labels).unsqueeze(1))

    def load_validation(self, oneshot: bool, batch_size: int = 1, shuffle: bool = False,
                        num_workers: int = 6) -> DataLoader:
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
            return DataLoader(val_dataset, batch_size, shuffle, num_workers=num_workers)

        else:
            print("Loading validation loader", flush=True)
            val_dataset = self._load_uniform_dataset_from_alphabets(self._eval_alphabet_dict, alphabets,
                                                                    self.validation_samples, writers)
            return DataLoader(val_dataset, batch_size, shuffle, num_workers=num_workers)

    def load_test(self, batch_size: int = 1, shuffle: bool = False, num_workers: int = 6) -> DataLoader:
        print("Loading oneshot test loader", flush=True)
        return DataLoader(self._load_test_dataset(), batch_size, shuffle, num_workers=num_workers)

    def _load_test_dataset(self):
        # alphabets 41-50
        # writers 17-20
        # uniform alphabet distribution

        alphabets = self._eval_alphabets[10:]  # Take only the last 10 alphabets for test
        writers = list(range(17, 20 + 1))  # Take only the last 4 writers

        # TODO should I add batch here
        return self._load_one_shot_dataset(self._eval_alphabet_dict, alphabets, self.test_shots, 20, writers)

    @staticmethod
    def _load_one_shot_dataset(alphabet_dict: Dict, alphabets: List[str], shots: int, n_way: int,
                               writer_range: List[int]):

        data = []
        for _ in tqdm(range(shots // n_way)):
            alphabet = random.choice(alphabets)
            while len(alphabet_dict[alphabet]) < n_way:
                alphabet = random.choice(alphabets)

            characters = random.sample(list(range(len(alphabet_dict[alphabet]))), n_way)
            drawer_1_idx, drawer_2_idx = random.sample(writer_range, 2)
            drawer_1_idx, drawer_2_idx = drawer_1_idx - 1, drawer_2_idx - 1

            data.append([[], []])
            for i in range(n_way):
                data[-1][0].append(
                    OmniglotDataLoaderCreator.TRANSFORMATIONS(
                        alphabet_dict[alphabet][characters[i]][drawer_1_idx]).unsqueeze(0))
                data[-1][1].append(
                    OmniglotDataLoaderCreator.TRANSFORMATIONS(
                        alphabet_dict[alphabet][characters[i]][drawer_2_idx]).unsqueeze(0))
            data[-1] = torch.cat([torch.cat(data[-1][0]).unsqueeze(0), torch.cat(data[-1][1]).unsqueeze(0)])
            data[-1] = data[-1].unsqueeze(0)

        return TensorDataset(torch.cat(data))

    @staticmethod
    def _load_alphabets_from_disk(path: str) -> (Dict, List):
        alphabet_dict = {}
        alphabets = sorted(os.listdir(path))

        ## white
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            OmniglotDataLoaderCreator.TRANSFORMATIONS
        ])

        ## black (better for RELU)
        # transforms = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor()
        # ])

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

                    ## white
                    # img = np.asarray(transforms(Image.open(image_location)))
                    img = Image.open(image_location).copy()

                    ## black
                    # img = 1 - np.asarray(transforms(Image.open(image_location)))
                    # img = (img - (1 - 0.9220603704452515)) / (0.2680765986442566)

                    alphabet_dict[alphabet][char_idx][writer_idx] = img

        return alphabets, alphabet_dict


class OmniglotVisualizer:

    @staticmethod
    def make_next_batch_grid(img_pairs):
        return torchvision.utils.make_grid([torch.cat([pair[0], pair[1]], dim=2) for pair in img_pairs], nrow=5,
                                           normalize=True)

    @staticmethod
    def make_next_oneshot_batch_grid(data):
        return torchvision.utils.make_grid([torch.cat([
            torchvision.utils.make_grid(pair[0], nrow=5, normalize=True),
            torchvision.utils.make_grid(pair[1], nrow=5, normalize=True)
        ], dim=2) for pair in data], nrow=1)

    @staticmethod
    def visualize_next_batch(loader, show_plot=True, save_plot=False):
        img_pairs, _ = next(iter(loader))
        grid = OmniglotVisualizer.make_next_batch_grid(img_pairs)
        OmniglotVisualizer.display(grid, show_plot, save_plot)

    @staticmethod
    def visualize_next_oneshot_batch(oneshot_loader, show_plot=True, save_plot=False):
        img_pairs, = next(iter(oneshot_loader))
        grid = OmniglotVisualizer.make_next_oneshot_batch_grid(img_pairs)
        OmniglotVisualizer.display(grid, show_plot, save_plot)

    @staticmethod
    def display(grid, show_plot=True, save_plot=False):
        plt.figure(figsize=(18, 24))
        plt.axis('off')
        plt.imshow(np.transpose(grid, (1, 2, 0)))

        if save_plot:
            plt.savefig(OmniglotVisualizer.generate_png_name())

        if show_plot:  # TODO does this make sense in matplotlib?
            plt.show()

    @staticmethod
    def generate_png_name():
        return OmniglotVisualizer.generate_name("png")

    @staticmethod
    def generate_name(extension="png"):
        return f"./{datetime.now()}.{extension}"


if __name__ == '__main__':
    random.seed(72)
    np.random.seed(72)
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Testing omniglot dataloader")
    omniglot_dataloader_creator = OmniglotDataLoaderCreator("./data/", 3000, 1000, 320, 400, False, 8)
    print("Preprocessing done.")
    print()

    print("Creating dataloaders:")
    # train_loader = omniglot_dataloader_creator.load_train(100, True)
    train_loader = omniglot_dataloader_creator.load_train(100, False)
    val_loader = omniglot_dataloader_creator.load_validation(False, 100, False)
    val_oneshot_loader = omniglot_dataloader_creator.load_validation(True, 5, False)
    test_oneshot_loader = omniglot_dataloader_creator.load_test(5, False)

    list(iter(train_loader.dataset))
    OmniglotVisualizer.visualize_next_batch(train_loader, False, True)

    list(iter(val_loader))
    list(iter(val_oneshot_loader))
    OmniglotVisualizer.visualize_next_batch(val_loader, False, True)
    OmniglotVisualizer.visualize_next_oneshot_batch(val_oneshot_loader, False, True)

    list(iter(test_oneshot_loader))
    OmniglotVisualizer.visualize_next_oneshot_batch(test_oneshot_loader, False, True)

    ## PLAYGROUND
    data, = next(iter(test_oneshot_loader))
    new_shape = (30, 1, 105, 105)
    ae = data[0][0][0].expand(new_shape)
    grid = torchvision.utils.make_grid(ae, nrow=5, normalize=True)
    plt.figure(figsize=(18, 18))
    plt.axis("off")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(OmniglotVisualizer.generate_name())
    # perfect!

    ## Lets calculate the norm and std of the train dataset
    all_pics = []
    for _, chars in omniglot_dataloader_creator._train_alphabet_dict.items():
        for _, writer in chars.items():
            for w_idx in range(12):
                all_pics.append(np.asarray(writer[w_idx], dtype=float))

    print(f"Train pics number is {len(all_pics)}. Chars: {len(all_pics) // 12}")
    all_pics = torch.tensor(all_pics)
    mean = all_pics.mean()
    std = all_pics.std()
    print(f"Mean is {mean}. Std is {std}")
    # Mean is 0.9220603704452515.Std is 0.2680765986442566 --> for all 20 writers - wrong!
    # Mean is 0.9217844519089332. Std is 0.26851047769786074

    ## After transformations
    all_pics = []
    for _, chars in omniglot_dataloader_creator._train_alphabet_dict.items():
        for _, writer in chars.items():
            for w_idx in range(12):
                all_pics.append(OmniglotDataLoaderCreator.TRANSFORMATIONS(np.asarray(writer[w_idx], dtype=float)).unsqueeze(0))
    all_pics = torch.cat(all_pics)
    mean = all_pics.mean()
    std = all_pics.std()
    print(f"AFTER TRANSFORMATIONS: Mean is {mean}. Std is {std}")
    # AFTER TRANSFORMATIONS: Mean is 4.543505702984737e-14. Std is 1.0000000010851275

    ## Plot whole train loader
    t = list(iter(train_loader))
    img_pairs = [e[0] for e in t]
    img_labels = [e[1] for e in t]
    # OmniglotVisualizer.visualize_next_batch([[img_pairs, img_labels]], True, True)
