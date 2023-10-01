from .nonadaptivedataloader import NonAdaptiveDSSDataLoader
from cords.selectionstrategies.SL import WeightedRandomExplorationStrategy
import numpy as np
import torch
from tqdm import tqdm
import time, copy
from transformers import (
    ViTFeatureExtractor,
    ViTModel,
    SegformerFeatureExtractor,
    SegformerModel,
    SegformerForSemanticSegmentation,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
)
from PIL import Image
from torch.utils.data import BatchSampler, SequentialSampler


class LeastConfidenceUncertainty(NonAdaptiveDSSDataLoader):
    """
    Fill in`.

    Parameters
    -----------
    train_loader: torch.utils.data.DataLoader class
        Dataloader of the training dataset
    dss_args: dict
        Data subset selection arguments dictionary required for CRAIG subset selection strategy
    logger: class
        Logger for logging the information
    """

    def __init__(self, train_loader, train_dataset, dss_args, logger, *args, **kwargs):
        """
        Arguments assertion check
        """
        assert "method" in dss_args.keys(), "'measure' is a compulsory argument. Include it as a key in dss_args"
        assert "model" in dss_args.keys(), "'model' is a compulsory argument. Include it as a key in dss_args"
        self.measure = dss_args.method
        self.model = dss_args.model
        self.dataset = train_dataset
        self.reverse = dss_args.reverse

        super(LeastConfidenceUncertainty, self).__init__(train_loader, train_loader, dss_args, logger, *args, **kwargs)

    def _init_subset_loader(self):
        """
        Function that initializes the initial subset loader
        """
        self.subset_indices, self.subset_weights = self._init_subset_indices()
        self._refresh_subset_loader()

    def _init_subset_indices(self):
        """
        Function that calls the CRAIG strategy for initial subset selection and calculating the initial subset weights.
        """
        start = time.time()
        self.logger.debug("Epoch: {0:d}, requires subset selection. ".format(self.cur_epoch))

        # load model
        model, feature_extractor = self._load_model()
        model.eval()

        # Generate predictions
        preds = []
        # with torch.no_grad():
        #     for idx, batch in tqdm(
        #         enumerate(self.train_loader),
        #         total=len(self.train_loader),
        #         desc="Determining uncertanties on training set",
        #     ):
        #         image, label, _, _ = batch
        #         image = image.cuda()
        #         label = label.long().cuda()

        #         if feature_extractor is not None:
        #             input = feature_extractor(image)
        #         else:
        #             input = image
        #         pred = model(input)
        #         preds.append(pred)
        images = []
        labels = []
        for x in tqdm(
            self.dataset,
            total=len(self.dataset),
            desc="loading dataset for least confidence uncertainty",
        ):
            images.append(Image.fromarray(np.transpose(x[0], (1, 2, 0)), mode="RGB"))
            labels.append(x[1])

        sampler = BatchSampler(SequentialSampler(range(len(images))), 10, drop_last=False)

        inputs = []
        for indices in sampler:
            if images[0].mode == "L":
                images_batch = [images[x].convert("RGB") for x in indices]
            else:
                images_batch = [images[x] for x in indices]
            inputs.append(feature_extractor(images_batch, return_tensors="pt"))

        img_preds = None  # Initialise

        # Warning large RAM requirements
        for batch_inputs in tqdm(inputs, total=len(inputs), desc="Extracting logits"):
            tmp_feat_dict = {}
            for key in batch_inputs.keys():
                tmp_feat_dict[key] = batch_inputs[key].to(device=self.device)
            with torch.no_grad():
                batch_outputs = model(**tmp_feat_dict)

            # TODO: Check this logic regarding dimension collapse for approximation
            if self.measure == "least_confidence":
                batch_preds = np.max(torch.softmax(batch_outputs.logits, 1).cpu().numpy(), axis=1)
            elif self.measure in "entropy":
                batch_preds = torch.softmax(batch_outputs.logits, 1).cpu().mean(axis=[2, 3])

            if img_preds is None:
                img_preds = batch_preds  # First iteration
            else:
                img_preds = np.append(img_preds, batch_preds, axis=0)  # Collect preds
            del tmp_feat_dict

        assert img_preds.shape[0] == len(images)

        # Least Confidence Uncertainty Sub-sampling
        if self.measure == "least_confidence":
            probs = img_preds.mean(axis=(1, 2))
            indices = probs.argsort(axis=0)
        elif self.measure in "entropy":
            entropy = (np.log(img_preds) * img_preds).sum(axis=1) * -1.0
            indices = entropy.argsort(axis=0)[::-1]
        else:
            raise NotImplementedError(f"'{self.measure}' method doesn't exist")

        if self.reverse:
            indices = np.flip(indices)

        subset_indices = indices[: self.budget]
        subset_weights = np.ones_like(subset_indices)  # All subsets have equal weight

        end = time.time()
        self.logger.info(
            "Epoch: {0:d}, Least Confidence Uncertainty sub-sampling subset selection finished, takes {1:.4f}. ".format(self.cur_epoch, (end - start))
        )
        return subset_indices, subset_weights

    def _load_model(
        self,
    ):
        if self.model == "segformer":
            model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            model = model.to(self.device)
        elif self.model == "deeplabv3":
            feature_extractor = AutoImageProcessor.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")
            model = AutoModelForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")
            model = model.to(self.device)
        else:
            raise NotImplementedError(f"{self.model} not implemented for Least Confidence Uncertainty Sub-sampling selection method")

        return model, feature_extractor
