import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/lib")
sys.path.insert(0, os.getcwd() + "/cords")

from cords.utils.data.dataloader.SL.adaptive.adaptivedataloader import AdaptiveDSSDataLoader

# from cords.selectionstrategies.SL import ClassWeightedRandomExplorationStrategy

from cords.utils.data.dataloader.SL.adaptive.adaptivedataloader import AdaptiveDSSDataLoader
from cords.utils.data.dataloader.SL.adaptive.adaptiverandomdataloader import AdaptiveRandomDataLoader
from cords.selectionstrategies.SL import RandomStrategy
import time
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

# Debugging
import torch
import logging
from dotmap import DotMap


from lib.datasets.pascal_ctx import PASCALContext


class PMWAdaptiveRandomDataLoader(AdaptiveRandomDataLoader):
    """
    Implements of PMWAdaptiveRandomDataLoader that serves as the dataloader for the pixel map weightedadaptive Random subset selection strategy.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary required for Random subset selection strategy
    logger: class
        Logger for logging the information
    """

    def __init__(
        self, train_loader, dss_args, logger, oracle: bool = False, base_set: bool = True, base_set_threshold: float = None, *args, **kwargs
    ):
        """
        Constructor function
        """
        super(PMWAdaptiveRandomDataLoader, self).__init__(train_loader=train_loader, dss_args=dss_args, logger=logger, *args, **kwargs)
        self.oracle = oracle
        self.base_set = base_set
        self.base_set_threshold = base_set_threshold
        self._init_class_weights()
        if self.base_set:
            assert self.base_set_threshold is not None, "Must define base set threshold"
            self._init_base_set()
            self.base_proportion = len(self.base_set_indices) / len(self.dataset)
            self.budget = self.budget - len(self.base_set_indices)

            # Lazy Inheritance
            self.subset_indices, self.subset_weights = self._resample_subset_indices()
            self._refresh_subset_loader()

    def _init_base_set(
        self,
    ):
        """
        Define the base set of indicies to be sampled at every epoch
        """
        class_indices = [c for c in self.class_proportions.keys() if self.class_proportions[c] < self.base_set_threshold]

        # Precense method: Include index if any under-represented classes are present.
        self.base_set_indices = []
        for i in self.indices:
            classes_present, _ = self.index_proportions[i]
            if np.isin(element=classes_present, test_elements=class_indices).any():
                self.base_set_indices.append(i)

    def _resample_subset_indices(self):
        """
        Function that calls the PMW Random subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        print("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))
        self.logger.debug("Random budget: %d", self.budget)
        subset_indices, subset_weights = self.strategy.select(self.budget)
        if self.base_set:
            subset_indices = np.hstack([subset_indices, self.base_set_indices])
            subset_weights = np.hstack([subset_weights, [1] * len(self.base_set_indices)])
        end = time.time()
        self.logger.info("Epoch: {0:d}, AdaptiveRandom subset selection finished, takes {1:.4f}. ".format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights

    def _init_class_weights(self):
        if self.oracle:
            # Oracle experiment defines class proportions according to the inverse of validation performance.
            performance_weights = [
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
                0.29,
                0.13,
            ]
            performance_weights = [p / sum(performance_weights) for p in performance_weights]
            self.class_class_weights = dict(zip(range(len(performance_weights)), performance_weights))
        else:
            uniques = []
            counts = []
            indices = []
            for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Calculating class proportions"):
                inputs, labels, size, names = batch
                for i in range(len(names)):
                    unique, count = np.unique(labels[i], return_counts=True)
                    uniques.append(unique)
                    counts.append(count)
                    index = self.train_loader.dataset.get_img_index(names[i])
                    indices.append(index)
                    assert names[i] == self.train_loader.dataset.__getitem__(index)[3], "Wrong index"
            pixel_df = pd.DataFrame({"class": np.concatenate(uniques), "count": np.concatenate(counts)})
            pixel_df = pixel_df.groupby("class").sum()
            pixel_df = pixel_df.drop(index=-1)
            pixel_df = pixel_df / pixel_df.sum()
            self.class_proportions = (pixel_df / pixel_df.sum())["count"].to_dict()
            self.index_proportions = dict(zip(indices, [u for u in zip(uniques, counts)]))
            self.indices = indices

    # def _index_to_class_prop(self, index: int) -> float:
    #     weight = self.class_proportions[index]
    #     return weight

    # def remap_values(self, x):
    #     # ala: https://discuss.pytorch.org/t/cv2-remap-in-pytorch/99354/10
    #     remapping = torch.arrange(0, self.num_classes), torch.Tensor(self.class_proportions.values)

    #     index = torch.bucketize(x.ravel(), remapping[0])
    #     return remapping[1][index].reshape(x.shape)

    # def get_pixel_map_weights(self, labels: torch.tensor) -> torch.tensor:
    #     weights = self._remap_values(labels)
    #     return weights


if __name__ == "__main__":
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
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        sampler=None,
    )

    # FIXME: Shuffle must be False

    logging.basicConfig(filename="", format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()

    dss_args = DotMap(dict(select_every=1, device="cpu", kappa=0, fraction=0.5))

    test_dataloader = PMWAdaptiveRandomDataLoader(
        train_loader=trainloader, dss_args=dss_args, logger=logger, oracle=False, base_set_threshold=0.003, base_set=True, batch_size=4
    )

    # Epoch 0
    for idx, batch in enumerate(test_dataloader):
        inputs, labels, sizes, names, sample_weights = batch
        x = None

    # Epoch 1
    for idx, batch in enumerate(test_dataloader):
        inputs, labels, sizes, names, sample_weights = batch
        x = None

    # Epoch 2
    for idx, batch in enumerate(test_dataloader):
        inputs, labels, sizes, names, sample_weights = batch
        x = None
