import numpy as np
import torch, time
import pickle
import random

# from torch.nn import Softmax
from torch.utils.data.dataloader import DataLoader

import math
from typing import Tuple
from tqdm import tqdm
import pandas as pd
import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/lib")

from lib.datasets.pascal_ctx import PASCALContext


class ClassWeightedRandomExplorationStrategy(object):
    """
    Implementation of thenovel Class Weighted Random Exploration Strategy class, where we select a set of points based on the proportions of samples
    in the training set..

    Parameters
    ----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    """

    def __init__(
        self,
        trainloader,
        # online=False,
        oracle_experiment=False
        # temperature=1,
        # per_class=False,
    ):
        """
        Constructor method
        """
        self.trainloader = trainloader

        # Allow for distributed dataloaders
        try:
            self.N_trn = len(trainloader.sampler.data_source)
        except:
            print("Distributed dataloader")
            self.N_trn = len(trainloader.dataset)

        self.class_proportions, self.index_proportions = self._get_proportions(self.trainloader)

        if oracle_experiment:
            # Oracle experiment defines class proportions according to the inverse of validation performance.
            performance_proportions = [
                0.77,
                0.13,
                0.08,
                0.24,
                0.02,
                0.7,
                0.75,
                0.57,
                0.35,
                0.7,
                0.5,
                0.8,
                0.27,
                0.78,
                0.79,
                0.46,
                0.35,
                0.11,
                0.21,
                0.55,
                0.29,
                0.35,
                0.75,
                0.11,
                0.3,
                0.58,
                0.14,
                0.22,
                0.73,
                0.44,
                0.65,
                0.61,
                0.37,
                0.73,
                0.42,
                0.15,
                0.8,
                0.15,
                0.34,
                0.39,
                0.41,
                0.25,
                0.61,
                0.19,
                0.13,
                0.26,
                0.91,
                0.44,
                0.32,
                0.47,
                0.58,
                0.75,
                0.73,
                0.2,
                0.7,
                0.58,
                0.7,
                0.29,
                0.13,
            ]

            performance_proportions = [p / sum(performance_proportions) for p in performance_proportions]
            self.class_proportions = dict(zip(range(len(performance_proportions)), performance_proportions))

    def _get_proportions(self, dataloader: DataLoader, mode: str = "image_wise") -> Tuple[dict, dict]:
        uniques = []
        counts = []
        indexes = []
        if mode == "pixel_wise":
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating class proportions"):
                inputs, labels, size, names = batch
                for i in range(len(names)):
                    unique, count = np.unique(labels[i], return_counts=True)
                    uniques.append(unique)
                    counts.append(count)
                    index = dataloader.dataset.get_img_index(names[i])
                    indexes.append(index)
                    assert names[i] == dataloader.dataset.__getitem__(index)[3], "Wrong index"
            pixel_df = pd.DataFrame({"class": np.concatenate(uniques), "count": np.concatenate(counts)})
            pixel_df = pixel_df.groupby("class").sum()
            pixel_df = pixel_df.drop(index=-1)
            pixel_df = pixel_df / pixel_df.sum()
            class_proportions = (pixel_df / pixel_df.sum())["count"].to_dict()
            index_proportions = dict(zip(indexes, [u for u in zip(uniques, counts)]))
            self.indexes = indexes
        elif mode == "image_wise":
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating class proportions"):
                inputs, labels, size, names = batch
                for i in range(len(names)):
                    img_class = dataloader.dataset.class_name_to_index(dataloader.dataset.get_imagewise_label(names[i]))
                    uniques.append(img_class)
                    counts.append(1)
                    index = dataloader.dataset.get_img_index(names[i])
                    indexes.append(index)
                    assert names[i] == dataloader.dataset.__getitem__(index)[3], "Wrong index"
            pixel_df = pd.DataFrame({"class": uniques, "count": counts})
            pixel_df = pixel_df.groupby("class").sum()
            pixel_df = pixel_df / pixel_df.sum()
            class_proportions = (pixel_df / pixel_df.sum())["count"].to_dict()
            index_proportions = dict(zip(indexes, [u for u in zip(uniques, counts)]))
            self.indexes = indexes
            # dataset_classes = [
            #     dataloader.dataset.class_name_to_index(dataloader.dataset.get_imagewise_label(file["file_name"])) for file in dataloader.dataset.files
            # ]
            # dataset_class_dist = {x: dataset_classes.count(x) for x in dataset_classes}
            # dataset_class_dist = OrderedDict(sorted(dataset_class_dist.items()))
            # image_df = pd.DataFrame.from_dict(data=dataset_class_dist, orient="index", columns=["count"])
            # class_proportions = (image_df / image_df.sum())["count"].to_dict()
            # index_proportions = None
        return class_proportions, index_proportions

    def _get_index_sampling_weights(self):
        PIXEL_WISE = False
        if PIXEL_WISE:
            weights = []
            for i in self.indexes:
                weight = 0
                classes, counts = self.index_proportions[i]
                for j in range(len(classes)):
                    if classes[j] == -1:  # Ignore background
                        pass
                    else:
                        weight = weight + (1 / self.class_proportions[classes[j]])  # * counts[j]
                # weight = weight / len(classes)
                weights.append(weight)
            # weights = [1 - (w / max(weights)) for w in weights]
        else:
            weights = []
            for i in self.indexes:
                weight = 0
                image_class_name = self.trainloader.dataset.get_imagewise_label(self.trainloader.dataset[i][3])
                image_class_index = self.trainloader.dataset.class_name_to_index(image_class_name)
                weight = 1 / self.class_proportions[image_class_index]
                weights.append(weight)
        return weights

    def select(self, budget):
        """
        Samples subset of size budget from the generated probability distribution.

        Parameters
        ----------
        budget: int
            The number of data points to be selected

        Returns
        ----------
        indices: ndarray
            Array of indices of size budget selected randomly
        gammas: Tensor
            Gradient weight values of selected indices
        """

        self.index_sampling_weights = self._get_index_sampling_weights()
        assert len(self.index_sampling_weights) == len(self.index_proportions)

        # FIXME: Are indicies from 0: len(trainloader) or are they the Pascal_ctx indexes?
        # WARNING: SELECTION WITH REPLACEMENT
        self.indices = random.choices(self.indexes, weights=self.index_sampling_weights, k=budget)

        self.gammas = torch.ones(len(self.indices))  # Dummy return

        return self.indices, self.gammas


if __name__ == "__main__":
    from tqdm import tqdm
    import seaborn as sns
    from collections import OrderedDict

    def jaccard_set(list1, list2):
        """Define Jaccard Similarity function for two sets"""
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    trainset = PASCALContext(
        root="/home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/data/",
        list_path="train",
        num_samples=None,
        num_classes=59,
        multi_scale=True,
        flip=True,
        ignore_label=-1,
        base_size=520,
        crop_size=(520, 520),
        downsample_rate=1,
        scale_factor=16,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        sampler=None,
    )

    p = 0.5
    strat = ClassWeightedRandomExplorationStrategy(trainloader, oracle_experiment=False)
    subsets = [strat.select(int(len(trainloader.dataset) * p))[0] for i in range(3)]
    pixel_dfs = []

    MODE = "IMAGE-WISE"

    if MODE == "PIXEL-WISE":
        # # PIXEL-WISE
        for subset in tqdm(subsets):
            uniques = []
            counts = []
            for index in subset:
                image, labels, shape, name = trainset.__getitem__(index)
                unique, count = np.unique(labels, return_counts=True)
                uniques.append(unique)
                counts.append(count)
            pixel_df = pd.DataFrame({"class": np.concatenate(uniques), "count": np.concatenate(counts)})
            pixel_df = pixel_df.groupby("class").sum()
            pixel_df = pixel_df.drop(index=-1)
            pixel_df = pixel_df / pixel_df.sum()
            pixel_dfs.append(pixel_df)
        total_df = pd.concat(pixel_dfs, axis=1)
        total_df["class_proportions/mIoU"] = strat.class_proportions.values()
    elif MODE == "IMAGE_OCCURENCE-WISE":
        # IMAGE-OCCURENCE-WISE
        for subset in tqdm(subsets):
            uniques = []
            counts = []
            for index in subset:
                image, labels, shape, name = trainset.__getitem__(index)
                unique, _ = np.unique(labels, return_counts=True)
                count = np.ones_like(unique)
                uniques.append(unique)
                counts.append(count)
            pixel_df = pd.DataFrame({"class": np.concatenate(uniques), "count": np.concatenate(counts)})
            pixel_df = pixel_df.groupby("class").sum()
            pixel_df = pixel_df.drop(index=-1)
            pixel_df = pixel_df / pixel_df.sum()
            pixel_dfs.append(pixel_df)
        total_df = pd.concat(pixel_dfs + [pd.DataFrame(strat.class_proportions.values())], axis=1)
    elif MODE == "IMAGE-WISE":
        # IMAGE-WISE
        class_distributions = []
        subset_dataframes = []
        i = 0
        for subset in tqdm(subsets):
            subset_classes = [trainset.class_name_to_index(trainset.get_imagewise_label(trainset[index][3])) for index in subset]
            subset_class_dist = {x: subset_classes.count(x) for x in subset_classes}
            subset_class_dist = OrderedDict(sorted(subset_class_dist.items()))
            class_distributions.append(subset_class_dist)
            subset_df = pd.DataFrame.from_dict(data=subset_class_dist, orient="index", columns=[f"subset_{i}"])
            subset_df = subset_df / subset_df.sum()
            i = i + 1
            subset_dataframes.append(subset_df)
        total_df = pd.concat(subset_dataframes + [pd.DataFrame(strat.class_proportions.values())], axis=1)
        # summary_df = pd.DataFrame({'mean':pd.concat(subset_dataframes, axis=1).mean(axis=1), 'var':pd.concat(subset_dataframes, axis=1).var(axis=1)})

    print(total_df.corr())
    total_df = total_df.set_axis(["subset_a", "subset_b", "subset_c", "trainset"], axis=1)

    plot_df = total_df.melt(var_name="set", value_name="count", ignore_index=False)
    plt = sns.barplot(data=plot_df, x=plot_df.index, y="count", hue="set")
    plt.set_xticklabels(plt.get_xticklabels(), rotation=90)
    plt.set_title("Mean Image-Occurence class distribution over data subsets, n=10")
    plt.set_ylabel("")

    # stats = []
    # n = 5
    # for _ in range(n):
    #     selection_a = strat.select(int(5000 * 0.5))[0]
    #     selection_b = strat.select(int(5000 * 0.5))[0]
    #     stats.append(jaccard_set(selection_a, selection_b))

    # stat = np.mean(stats)
    # print(f"Mean Jaccard Similarity for selected subsets is: ")
