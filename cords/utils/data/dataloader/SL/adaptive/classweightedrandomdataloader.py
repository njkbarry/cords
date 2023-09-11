import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd() + "/lib")
sys.path.insert(0, os.getcwd() + "/cords")

from cords.utils.data.dataloader.SL.adaptive.adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import ClassWeightedRandomExplorationStrategy
import time

import torch
import logging
from dotmap import DotMap


from lib.datasets.pascal_ctx import PASCALContext


class ClassWeightedRandomDataLoader(AdaptiveDSSDataLoader):
    """
    Implements of ClassWeightedRandomDataLoader
    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary required for CRAIG subset selection strategy
    logger: class
        Logger for logging the information
    """

    def __init__(self, train_loader, dss_args, logger, oracle_experiment, *args, **kwargs):
        self.strategy = ClassWeightedRandomExplorationStrategy(train_loader, oracle_experiment=oracle_experiment)
        super(ClassWeightedRandomDataLoader, self).__init__(train_loader, train_loader, dss_args, logger, *args, **kwargs)
        """
         Arguments assertion check
        """
        # assert "global_order_file" in dss_args.keys(), "'global_order_file' is a compulsory argument. Include it as a key in dss_args"
        # assert "temperature" in dss_args.keys(), "'temperature' is a compulsory argument. Include it as a key in dss_args"
        # assert "per_class" in dss_args.keys(), "'per_class' is a compulsory argument. Include it as a key in dss_args"

        self.logger.debug("Global order dataloader initialized.")

    # Over-riding initial random subset selection
    def _init_subset_loader(self):
        """
        Function that initializes the initial subset loader
        """
        self.subset_indices, self.subset_weights = self._init_subset_indices()
        self._refresh_subset_loader()

    def _init_subset_indices(self):
        """
        Function that initializes the initial subset indices
        """
        start = time.time()
        self.logger.debug("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))
        subset_indices, subset_weights = self.strategy.select(self.budget)
        end = time.time()
        self.logger.info("Epoch: {0:d}, ClassProportion based subset selection finished, takes {1:.4f}. ".format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights

    def _resample_subset_indices(self):
        """
        Function that calls the Weighted Random subset selection strategy to sample new subset indices and the corresponding subset weights.
        """
        start = time.time()
        print("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))
        self.logger.debug("ClassProportion budget: %d", self.budget)
        subset_indices, subset_weights = self.strategy.select(self.budget)
        end = time.time()
        self.logger.info("Epoch: {0:d}, ClassProportion based subset selection finished, takes {1:.4f}. ".format(self.cur_epoch, (end - start)))
        return subset_indices, subset_weights


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

    cwrdl = ClassWeightedRandomDataLoader(train_loader=trainloader, dss_args=dss_args, logger=logger, oracle_experiment=True, batch_size=4)

    for idx, batch in enumerate(cwrdl):
        inputs, labels, sizes, names, sample_weights = batch
        x = None

    for idx, batch in enumerate(trainloader):
        inputs, labels, sizes, names = batch
        x = None
