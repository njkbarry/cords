import datasets
import torchvision
from sentence_transformers import SentenceTransformer, util
from matplotlib import pyplot as plt
from sklearn import metrics

from lib.models.seg_hrnet_ocr import get_seg_model
from ..datasets.__utils import TinyImageNet
from transformers import (
    ViTFeatureExtractor,
    ViTModel,
    SegformerFeatureExtractor,
    SegformerModel,
)
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, chi2
from ot import (
    sliced_wasserstein_distance,
    gromov_wasserstein,
    sinkhorn2,
    gromov_wasserstein2,
)
from numpy import linalg as LA
from sklearn.metrics.pairwise import euclidean_distances
from numba import jit, config
from torch.utils.data import random_split, BatchSampler, SequentialSampler
import torch
import pickle
import math
import os
import submodlib
import argparse
import h5py
import numpy as np
import time
from PIL import Image

from tqdm import tqdm
from multiprocessing import Pool, Process
from multiprocessing import freeze_support
import itertools

# Oracle loading
import lib.models
import lib.datasets
from lib.config import config, update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
import torch.nn as nn


def get_cdist_2D(matrix):
    from numpy import linalg as LA

    d = matrix.shape[0]
    cdist = np.zeros((d, d))
    for i in tqdm(range(d), total=d, desc="generating cdist 2D matrix"):
        for j in range(d):
            cdist[i][j] = LA.norm(matrix[i] - matrix[j], ord=2)
    return cdist


def get_sliced_wasserstein_dist(params):
    i, j = params
    return (i, j), sliced_wasserstein_distance(
        emb[i] / np.max(emb[i]), emb[j] / np.max(emb[j]), n_projections=100
    )


def get_gromov_wasserstein_dist(params):
    i, j = params
    dist = abs(gromov_wasserstein2(emb[i] / np.max(emb[i]), emb[j] / np.max(emb[j])))
    return (i, j), dist


def get_fronerbius_dist(params):
    i, j = params
    dist = LA.norm(emb[i] - emb[j], ord="fro")
    return (i, j), dist


def get_mmd_dist(params):
    i, j = params
    dist = mmd_rbf(emb[i] / np.max(emb[i]), emb[j] / np.max(emb[j]), gamma=gamma)
    return (i, j), dist


def mmd_rbf(X, Y, gamma=1.0):
    """
    ala: https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py

    NOTE:
        - define gamma according to suggestion in: https://torchdrift.org/notebooks/note_on_mmd.html

    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


WASSERSTEIN_METHOD_DICT = {
    "sliced_wasserstein": get_sliced_wasserstein_dist,
    "gromov_wasserstein": get_gromov_wasserstein_dist,
}


LABEL_MAPPINGS = {
    "glue_sst2": "label",
    "trec6": "coarse_label",
    "imdb": "label",
    "rotten_tomatoes": "label",
    "tweet_eval": "label",
}

SENTENCE_MAPPINGS = {
    "glue_sst2": "sentence",
    "trec6": "text",
    "imdb": "text",
    "rotten_tomatoes": "text",
    "tweet_eval": "text",
}

IMAGE_MAPPINGS = {"cifar10": "images"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Global ordering of a dataset using pretrained LMs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="glue_sst2",
        help="Only supports datasets for hugging face currently.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-distilroberta-v1",
        help="Transformer model used for computing the embeddings.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        help="Directory in which data downloads happen.",
        default="/home/krishnateja/data",
    )
    parser.add_argument(
        "--submod_function",
        type=str,
        default="logdet",
        help="Submdular function used for finding the global order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed value for reproducibility of the experiments.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device used for computing the embeddings",
    )
    args = parser.parse_args()
    return args


def dict2pickle(file_name, dict_object):
    """
    Store dictionary to pickle file
    """
    with open(file_name, "wb") as fOut:
        pickle.dump(dict_object, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def pickle2dict(file_name, key):
    """
    Load dictionary from pickle file
    """
    with open(file_name, "rb") as fIn:
        stored_data = pickle.load(fIn)
        value = stored_data[key]
    return value


def store_embeddings(pickle_name, embeddings):
    """
    Store embeddings to disc
    """
    with open(pickle_name, "wb") as fOut:
        pickle.dump({"embeddings": embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def load_embeddings(pickle_name):
    """
    Load embeddings from disc
    """
    with open(pickle_name, "rb") as fIn:
        stored_data = pickle.load(fIn)
        # stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data["embeddings"]
    return stored_embeddings


def get_cdist(V):
    ct = time.time()
    dist_mat = euclidean_distances(V)
    print("Distance Matrix construction time ", time.time() - ct)
    return get_square(dist_mat)


# @torch.jit.script
@jit(nopython=True, parallel=True)
def get_square(mat):
    return mat**2


@jit(nopython=True, parallel=True)
def get_rbf_kernel(dist_mat, kw=0.1):
    sim = np.exp(-dist_mat / (kw * dist_mat.mean()))
    return sim


# @jit(nopython=True, parallel=True)
def get_dot_product(mat):
    sim = np.matmul(mat, np.transpose(mat))
    return sim


def get_wasserstein(mat, n_projections=1000, method="sliced"):
    assert len(mat.shape) == 3, "embeddings are not 2D for wasserstein"

    global n
    global sim
    global emb

    n = int(mat.shape[0])
    sim = np.zeros((n, n))
    emb = mat

    n_processors = 8

    i = range(n)
    j = range(n)

    paramlist = list(itertools.product(i, j))

    pool = Pool(maxtasksperchild=100, processes=n_processors)

    similarities = {}

    func = WASSERSTEIN_METHOD_DICT[method]
    for k, v in tqdm(
        pool.imap_unordered(func, paramlist, chunksize=10000),
        total=len(paramlist),
    ):
        similarities[k] = v

    # for params in paramlist:
    #     k, v = get_gromov_wasserstein_dist(params)
    #     similarities[k] = v

    for k in similarities.keys():
        i, j = k
        sim[i][j] = similarities[k]

    sim = sim / np.max(sim)  # normalise

    return sim


def get_fronerbius(mat):
    assert len(mat.shape) == 3, "embeddings are not 2D for wasserstein"

    global n
    global sim
    global emb

    n = int(mat.shape[0])
    sim = np.zeros((n, n))
    emb = mat

    n_processors = 8

    i = range(n)
    j = range(n)

    paramlist = list(itertools.product(i, j))

    pool = Pool(maxtasksperchild=100, processes=n_processors)

    similarities = {}

    for k, v in tqdm(
        pool.imap_unordered(get_fronerbius_dist, paramlist, chunksize=10000),
        total=len(paramlist),
    ):
        similarities[k] = v

    for k in similarities.keys():
        i, j = k
        sim[i][j] = similarities[k]

    sim = sim / np.max(sim)  # normalise

    return sim


def get_mmd(mat):
    assert len(mat.shape) == 3, "embeddings are not 2D for wasserstein"

    global n
    global sim
    global emb
    global gamma

    n = int(mat.shape[0])
    sim = np.zeros((n, n))
    emb = mat

    n_processors = 8

    i = range(n)
    j = range(n)

    paramlist = list(itertools.product(i, j))

    pool = Pool(maxtasksperchild=100, processes=n_processors)

    similarities = {}

    gamma = np.median(get_cdist_2D(mat)) / 2

    for k, v in tqdm(
        pool.imap_unordered(get_mmd_dist, paramlist, chunksize=10000),
        total=len(paramlist),
    ):
        similarities[k] = v

    for k in similarities.keys():
        i, j = k
        sim[i][j] = similarities[k]

    sim = sim / np.max(sim)  # normalise

    return sim


def compute_text_embeddings(model_name, sentences, device, return_tensor=False):
    """
    Compute sentence embeddings using a transformer model and return in numpy or tensor format
    """
    model = SentenceTransformer(model_name, device=device)
    if return_tensor:
        embeddings = model.encode(
            sentences, device=device, convert_to_tensor=True
        ).cpu()
    else:
        embeddings = model.encode(sentences, device=device, convert_to_numpy=True)
    return embeddings


def compute_oracle_image_embeddings(
    images, device, return_tensor=False, mode="oracle_spat"
):
    def parse_model_args():
        parser = argparse.ArgumentParser(description="Train segmentation network")

        parser.add_argument(
            "--cfg", help="experiment configure file name", required=True, type=str
        )
        parser.add_argument("--seed", type=int, default=304)
        parser.add_argument("--local_rank", type=int, default=-1)
        parser.add_argument(
            "opts",
            help="Modify config options using the command-line",
            default=None,
            nargs=argparse.REMAINDER,
        )

        args = parser.parse_args()
        update_config(config, args)

        return args

    # generate config
    parse_model_args()

    # FIXME: Model type hard coded
    model = get_seg_model(config)

    # FIXME: final_output_dir hard coded
    final_output_dir = "/home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/output/proxy_experiment/oracle_model"

    # Load oracle weights
    model_state_file = os.path.join(final_output_dir, "checkpoint.pth.tar")

    checkpoint = torch.load(model_state_file, map_location={"cuda:0": "cpu"})

    if os.path.isfile(model_state_file):
        model.load_state_dict(
            {
                k.replace("model.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("model.")
            }
        )
    else:
        raise ValueError

    model.eval()
    model.to(device=device)

    sampler = BatchSampler(SequentialSampler(range(len(images))), 4, drop_last=False)

    img_features = []
    for indices in sampler:
        if images[0].mode == "L":
            images_batch = [images[x].convert("RGB") for x in indices]
        else:
            images_batch = [images[x] for x in indices]

        images_batch = [
            torch.from_numpy(np.array(img))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device)
            for img in images_batch
        ]

        images_batch = torch.cat(images_batch, dim=0)
        images_batch = images_batch.float()

        with torch.no_grad():
            if mode == "oracle_spat":
                img_features.append(
                    model.spatial_embed_input(images_batch).mean(dim=1).cpu()
                )
            elif mode == "oracle_context":
                img_features.append(
                    model.context_embed_input(images_batch).mean(dim=1).cpu()
                )
            elif mode == "oracle_feature_flat":
                img_features.append(
                    model.feature_embed_input(images_batch, flat=True).mean(dim=1).cpu()
                )
            else:
                raise NotImplementedError

        del images_batch

    img_features = torch.cat(img_features, dim=0)

    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_segformer_image_embeddings(images, device, return_tensor=False):
    # TODO: Ensure this does what you think it does
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = model.to(device)
    # inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))), 20, drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == "L":
            images_batch = [images[x].convert("RGB") for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state.mean(dim=1).cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict

    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_image_embeddings(model_name, images, device, return_tensor=False):
    """
    Compute image embeddings using CLIP based model and return in numpy or tensor format
    """
    model = SentenceTransformer(model_name, device=device)
    if return_tensor:
        embeddings = model.encode(images, device=device, convert_to_tensor=True).cpu()
    else:
        embeddings = model.encode(images, device=device, convert_to_numpy=True)
    return embeddings


def compute_vit_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-large-patch16-224-in21k"
    )
    model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
    model = model.to(device)
    # inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))), 20, drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == "L":
            images_batch = [images[x].convert("RGB") for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state.mean(
            dim=1
        ).cpu()  # Averages over channels
        img_features.append(batch_img_features)
        del tmp_feat_dict

    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_vit_cls_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-large-patch16-224-in21k"
    )
    model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
    model = model.to(device)
    # inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))), 20, drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == "L":
            images_batch = [images[x].convert("RGB") for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state[:, 0, :].cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict

    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_dino_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vitb16")
    model = ViTModel.from_pretrained("facebook/dino-vitb16")
    model = model.to(device)
    # inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))), 20, drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == "L":
            images_batch = [images[x].convert("RGB") for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state.mean(dim=1).cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict

    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_dino_cls_image_embeddings(images, device, return_tensor=False):
    feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vitb16")
    model = ViTModel.from_pretrained("facebook/dino-vitb16")
    model = model.to(device)
    # inputs = feature_extractor(images, return_tensors="pt")
    sampler = BatchSampler(SequentialSampler(range(len(images))), 20, drop_last=False)

    inputs = []
    for indices in sampler:
        if images[0].mode == "L":
            images_batch = [images[x].convert("RGB") for x in indices]
        else:
            images_batch = [images[x] for x in indices]
        inputs.append(feature_extractor(images_batch, return_tensors="pt"))

    img_features = []
    for batch_inputs in inputs:
        tmp_feat_dict = {}
        for key in batch_inputs.keys():
            tmp_feat_dict[key] = batch_inputs[key].to(device=device)
        with torch.no_grad():
            batch_outputs = model(**tmp_feat_dict)
        batch_img_features = batch_outputs.last_hidden_state[:, 0, :].cpu()
        img_features.append(batch_img_features)
        del tmp_feat_dict

    img_features = torch.cat(img_features, dim=0)
    if return_tensor == False:
        return img_features.numpy()
    else:
        return img_features


def compute_global_ordering(
    embeddings,
    submod_function,
    train_labels,
    kw,
    r2_coefficient,
    knn,
    metric,
    partition_mode,
    train_dataset,
):
    """
    Return greedy ordering and gains with different submodular functions as the global order.
    """

    # Partition methods for Semantic segmentation datasets
    train_labels = partition_dataset(
        train_labels=train_labels,
        partition_mode=partition_mode,
        train_dataset=train_dataset,
    )
    if submod_function not in [
        "supfl",
        "gc_pc",
        "logdet_pc",
        "disp_min_pc",
        "disp_sum_pc",
    ]:
        if metric in ["rbf_kernel", "dot", "cossim"]:  # 1D embedding vector
            if len(embeddings.shape) == 3:
                embeddings = embeddings.reshape(
                    [embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]]
                )

            data_dist = get_cdist(embeddings)

            if metric == "rbf_kernel":
                data_sijs = get_rbf_kernel(data_dist, kw)
            elif metric == "dot":
                data_sijs = get_dot_product(embeddings)
                if submod_function in ["disp_min", "disp_sum"]:
                    data_sijs = (data_sijs - np.min(data_sijs)) / (
                        np.max(data_sijs) - np.min(data_sijs)
                    )
                else:
                    if np.min(data_sijs) < 0:
                        data_sijs = data_sijs - np.min(data_sijs)
            elif metric == "cossim":
                normalized_embeddings = embeddings / np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )
                data_sijs = get_dot_product(normalized_embeddings)
                if submod_function in ["disp_min", "disp_sum"]:
                    data_sijs = (data_sijs - np.min(data_sijs)) / (
                        np.max(data_sijs) - np.min(data_sijs)
                    )
                else:
                    data_sijs = (data_sijs + 1) / 2
        elif metric in [
            "sliced_wasserstein",
            "gromov_wasserstein",
            "mmd",
            "chi",
            "fronerbius",
        ]:
            data_dist = get_cdist_2D(embeddings)

            if metric in ["sliced_wasserstein", "gromov_wasserstein"]:
                data_sijs = get_wasserstein(embeddings, method=metric)
            elif metric == "fronerbius":
                data_sijs = get_fronerbius(embeddings)
            elif metric == "mmd":
                data_sijs = get_mmd(embeddings)
            elif metric == "chi":
                raise NotImplementedError
        else:
            raise ValueError("Please enter a valid metric")

        data_knn = np.argsort(data_dist, axis=1)[:, :knn].tolist()
        data_r2 = np.nonzero(
            data_dist <= max(1e-5, data_dist.mean() - r2_coefficient * data_dist.std())
        )
        data_r2 = zip(data_r2[0].tolist(), data_r2[1].tolist())
        data_r2_dict = {}
        for x in data_r2:
            if x[0] in data_r2_dict.keys():
                data_r2_dict[x[0]].append(x[1])
            else:
                data_r2_dict[x[0]] = [x[1]]

    if submod_function == "fl":
        obj = submodlib.FacilityLocationFunction(
            n=embeddings.shape[0], separate_rep=False, mode="dense", sijs=data_sijs
        )

    elif submod_function == "logdet":
        obj = submodlib.LogDeterminantFunction(
            n=embeddings.shape[0], mode="dense", lambdaVal=1, sijs=data_sijs
        )

    elif submod_function == "gc":
        obj = submodlib.GraphCutFunction(
            n=embeddings.shape[0],
            mode="dense",
            lambdaVal=1,
            separate_rep=False,
            ggsijs=data_sijs,
        )

    elif submod_function == "disp_min":
        obj = submodlib.DisparityMinFunction(
            n=embeddings.shape[0], mode="dense", sijs=data_sijs
        )

    elif submod_function == "disp_sum":
        obj = submodlib.DisparitySumFunction(
            n=embeddings.shape[0], mode="dense", sijs=data_sijs
        )

    if submod_function in ["gc", "fl", "logdet", "disp_min", "disp_sum"]:
        if submod_function == "disp_min":
            greedyList = obj.maximize(
                budget=embeddings.shape[0] - 1,
                optimizer="NaiveGreedy",
                stopIfZeroGain=False,
                stopIfNegativeGain=False,
                verbose=False,
            )
        else:
            greedyList = obj.maximize(
                budget=embeddings.shape[0] - 1,
                optimizer="LazyGreedy",
                stopIfZeroGain=False,
                stopIfNegativeGain=False,
                verbose=False,
            )
        rem_elem = list(
            set(range(embeddings.shape[0])).difference(set([x[0] for x in greedyList]))
        )[0]
        rem_gain = greedyList[-1][1]
        greedyList.append((rem_elem, rem_gain))
    else:
        print(
            "WARNING PARTITION CLUSTER MILO CONFIGURATION (DO NOT USE 2D EMBEDDINGS HERE)"
        )
        # raise NotImplementedError
        clusters = set(train_labels)
        data_knn = [[] for _ in range(len(train_labels))]
        data_r2_dict = {}
        greedyList = []
        # Label-wise Partition
        cluster_idxs = {}
        for i in range(len(train_labels)):
            if train_labels[i] in cluster_idxs.keys():
                cluster_idxs[train_labels[i]].append(i)
            else:
                cluster_idxs[train_labels[i]] = [i]
        for cluster in clusters:
            idxs = cluster_idxs[cluster]
            cluster_embeddings = embeddings[idxs, :]

            print(cluster_embeddings.shape)
            clustered_dist = get_cdist(cluster_embeddings)
            if metric == "rbf_kernel":
                clustered_sijs = get_rbf_kernel(clustered_dist, kw)
            elif metric == "dot":
                clustered_sijs = get_dot_product(cluster_embeddings)
                if submod_function in ["disp_min_pc", "disp_sum_pc"]:
                    clustered_sijs = (clustered_sijs - np.min(clustered_sijs)) / (
                        np.max(clustered_sijs) - np.min(clustered_sijs)
                    )
                else:
                    if np.min(clustered_sijs) < 0:
                        clustered_sijs = clustered_sijs - np.min(clustered_sijs)
            elif metric == "cossim":
                normalized_embeddings = cluster_embeddings / np.linalg.norm(
                    cluster_embeddings, axis=1, keepdims=True
                )
                clustered_sijs = get_dot_product(normalized_embeddings)
                if submod_function in ["disp_min_pc", "disp_sum_pc"]:
                    clustered_sijs = (clustered_sijs - np.min(clustered_sijs)) / (
                        np.max(clustered_sijs) - np.min(clustered_sijs)
                    )
                else:
                    clustered_sijs = (1 + clustered_sijs) / 2
            else:
                raise ValueError("Please enter a valid metric")

            if submod_function in ["supfl"]:
                obj = submodlib.FacilityLocationFunction(
                    n=cluster_embeddings.shape[0],
                    separate_rep=False,
                    mode="dense",
                    sijs=clustered_sijs,
                )
            elif submod_function in ["gc_pc"]:
                obj = submodlib.GraphCutFunction(
                    n=cluster_embeddings.shape[0],
                    mode="dense",
                    lambdaVal=0.4,
                    separate_rep=False,
                    ggsijs=clustered_sijs,
                )
            elif submod_function in ["logdet_pc"]:
                obj = submodlib.LogDeterminantFunction(
                    n=cluster_embeddings.shape[0],
                    mode="dense",
                    lambdaVal=1,
                    sijs=clustered_sijs,
                )

            elif submod_function == "disp_min_pc":
                obj = submodlib.DisparityMinFunction(
                    n=cluster_embeddings.shape[0], mode="dense", sijs=clustered_sijs
                )

            elif submod_function == "disp_sum_pc":
                obj = submodlib.DisparitySumFunction(
                    n=cluster_embeddings.shape[0], mode="dense", sijs=clustered_sijs
                )

            if submod_function == "disp_min_pc":
                clustergreedyList = obj.maximize(
                    budget=cluster_embeddings.shape[0] - 1,
                    optimizer="NaiveGreedy",
                    stopIfZeroGain=False,
                    stopIfNegativeGain=False,
                    verbose=False,
                )
            else:
                clustergreedyList = obj.maximize(
                    budget=cluster_embeddings.shape[0] - 1,
                    optimizer="LazyGreedy",
                    stopIfZeroGain=False,
                    stopIfNegativeGain=False,
                    verbose=False,
                )
            rem_elem = list(
                set(range(cluster_embeddings.shape[0])).difference(
                    set([x[0] for x in clustergreedyList])
                )
            )[0]
            rem_gain = clustergreedyList[-1][1]
            clustergreedyList.append((rem_elem, rem_gain))
            clusteredgreedylist_with_orig_idxs = [
                (idxs[x[0]], x[1]) for x in clustergreedyList
            ]
            greedyList.extend(clusteredgreedylist_with_orig_idxs)
            del obj
            clustered_knn = np.argsort(clustered_dist, axis=1)[:, :knn].tolist()
            for i in range(len(idxs)):
                data_knn[idxs[i]] = [idxs[j] for j in clustered_knn[i]]
            clustered_r2 = np.nonzero(
                clustered_dist
                <= max(
                    1e-5, clustered_dist.mean() - r2_coefficient * clustered_dist.std()
                )
            )
            clustered_r2 = zip(clustered_r2[0].tolist(), clustered_r2[1].tolist())
            for x in clustered_r2:
                if idxs[x[0]] in data_r2_dict.keys():
                    data_r2_dict[idxs[x[0]]].append(idxs[x[1]])
                else:
                    data_r2_dict[idxs[x[0]]] = [idxs[x[1]]]
        greedyList.sort(key=lambda x: x[1], reverse=True)

    knn_list = []
    r2_list = []
    for x in greedyList:
        knn_list.append(data_knn[x[0]])
        r2_list.append(data_r2_dict[x[0]])
    # Sorted Label-wise Partition
    cluster_idxs = {}
    greedy_idxs = [x[0] for x in greedyList]

    for i in greedy_idxs:
        if train_labels[i] in cluster_idxs.keys():
            cluster_idxs[train_labels[i]].append(i)
        else:
            cluster_idxs[train_labels[i]] = [i]
    return greedyList, knn_list, r2_list, cluster_idxs


def partition_dataset(train_labels, partition_mode: str, train_dataset):
    # TODO: Create enum
    # Partition methods for Semantic segmentation datasets
    if partition_mode == "no_partition":
        print("DEV EXPERIMENT: NO PARTITIONING!")
        train_labels = [0] * len(train_labels)
    elif partition_mode == "pixel_mode":
        # Lazily approximate image-wise label based on most frequent segmentation mask vlaue.
        # Reasonable approximation in ImageNet style images consistent with pascal_ctx
        print("DEV EXPERIMENT: IMAGE-WISE PARTITIONING!")
        tmp = []
        for mask in train_labels:
            values, counts = np.unique(mask, return_counts=True)
            index = np.argmax(counts)
            tmp.append(index)
        train_labels = tmp
    elif partition_mode == "occurence_class_proportion":
        """
        Find partition by the least frequently occuring class that is present in the image
        """
        print("DEV EXPERIMENT: COCCURENCE-CLASS PARTITIONING!")
        tmp = []
        for mask in train_labels:
            values, counts = np.unique(mask, return_counts=True)
            try:
                index = np.nanargmin(
                    [train_dataset.get_occurence_class_proportion(c) for c in values]
                )
                # FIXME: Probably redundant, comment out for now
                # if values[index] == train_dataset.ignore_label:
                #     tmp.append(4)  # arbitrary
                tmp.append(values[index])
            except ValueError as e:
                print('Pure ignore image encountered')
                tmp.append(4)   # arbitrary
        train_labels = tmp

    elif partition_mode == "pascal_image_label":
        # Get the image-wise label:
        #   - When there are multiple for an image return the most frequent
        #   - All train_labels then are on the same order of magnitude.
        train_labels = [
            train_dataset.get_imagewise_label(train_dataset.__getitem__(i)[3])
            for i in range(len(train_dataset))
        ]
    elif partition_mode == "native":
        pass
    else:
        raise NotImplementedError()
    return train_labels


def compute_stochastic_greedy_subsets(
    embeddings,
    submod_function,
    train_labels,
    kw,
    metric,
    fraction,
    n_subsets=300,
    partition_mode="native",
    train_dataset=None,
    epsilon=0.001,
):  # Drunk hack
    """
    Return greedy ordering and gains with different submodular functions as the global order.
    """

    train_labels = partition_dataset(
        train_labels=train_labels,
        partition_mode=partition_mode,
        train_dataset=train_dataset,
    )

    budget = int(fraction * embeddings.shape[0])
    # if submod_function not in ["supfl", "gc_pc", "logdet_pc", "disp_min", "disp_sum"]:
    if submod_function not in ["supfl", "gc_pc", "logdet_pc"]:  # FIXME: Test code
        # TODO: Fix in more sensible place (segformer train embeddings generation)
        if metric in ["rbf_kernel", "dot", "cossim"]:  # 1D embedding vector
            if len(embeddings.shape) == 3:
                embeddings = embeddings.reshape(
                    [embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]]
                )

            data_dist = get_cdist(embeddings)

            if metric == "rbf_kernel":
                data_sijs = get_rbf_kernel(data_dist, kw)
            elif metric == "dot":
                data_sijs = get_dot_product(embeddings)
                if submod_function in ["disp_min", "disp_sum"]:
                    data_sijs = (data_sijs - np.min(data_sijs)) / (
                        np.max(data_sijs) - np.min(data_sijs)
                    )
                else:
                    if np.min(data_sijs) < 0:
                        data_sijs = data_sijs - np.min(data_sijs)
            elif metric == "cossim":
                normalized_embeddings = embeddings / np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                )
                data_sijs = get_dot_product(normalized_embeddings)
                if submod_function in ["disp_min", "disp_sum"]:
                    data_sijs = (data_sijs - np.min(data_sijs)) / (
                        np.max(data_sijs) - np.min(data_sijs)
                    )
                else:
                    data_sijs = (data_sijs + 1) / 2
        elif metric in [
            "sliced_wasserstein",
            "gromov_wasserstein",
            "mmd",
            "chi",
            "fronerbius",
        ]:
            if metric in ["sliced_wasserstein", "gromov_wasserstein"]:
                data_sijs = get_wasserstein(embeddings, method=metric)
            elif metric == "fronerbius":
                data_sijs = get_fronerbius(embeddings)
            elif metric == "mmd":
                data_sijs = get_mmd(embeddings)
            elif metric == "chi":
                raise NotImplementedError
        else:
            raise ValueError("Please enter a valid metric")

    if submod_function == "fl":
        obj = submodlib.FacilityLocationFunction(
            n=embeddings.shape[0], separate_rep=False, mode="dense", sijs=data_sijs
        )

    elif submod_function == "logdet":
        obj = submodlib.LogDeterminantFunction(
            n=embeddings.shape[0], mode="dense", lambdaVal=1, sijs=data_sijs
        )

    elif submod_function == "gc":
        obj = submodlib.GraphCutFunction(
            n=embeddings.shape[0],
            mode="dense",
            lambdaVal=1,
            separate_rep=False,
            ggsijs=data_sijs,
        )

    elif submod_function == "disp_min":
        obj = submodlib.DisparityMinFunction(
            n=embeddings.shape[0], mode="dense", sijs=data_sijs
        )

    elif submod_function == "disp_sum":
        obj = submodlib.DisparitySumFunction(
            n=embeddings.shape[0], mode="dense", sijs=data_sijs
        )

    subsets = []
    total_time = 0
    if submod_function not in [
        "supfl",
        "gc_pc",
        "logdet_pc",
        "disp_min_pc",
        "disp_sum_pc",
    ]:
        for _ in range(n_subsets):
            st_time = time.time()
            if submod_function == "disp_min":
                subset = obj.maximize(
                    budget=budget,
                    optimizer="StochasticGreedy",
                    epsilon=epsilon,
                    stopIfZeroGain=False,
                    stopIfNegativeGain=False,
                    verbose=False,
                )
            else:
                subset = obj.maximize(
                    budget=budget,
                    optimizer="LazierThanLazyGreedy",
                    epsilon=epsilon,
                    stopIfZeroGain=False,
                    stopIfNegativeGain=False,
                    verbose=False,
                )
            subsets.append(subset)
            total_time += time.time() - st_time
    else:
        # raise NotImplementedError("Epsilon from config not propogated to per cluster stochastic subset generation")
        print("WARNING PARTITION CLUSTER MILO CONFIGURATION")

        clusters = set(train_labels)
        # Label-wise Partition
        cluster_idxs = {}
        # print(train_labels)
        for i in range(len(train_labels)):
            if train_labels[i] in cluster_idxs.keys():
                cluster_idxs[train_labels[i]].append(i)
            else:
                cluster_idxs[train_labels[i]] = [i]

        per_cls_cnt = [len(cluster_idxs[key]) for key in cluster_idxs.keys()]
        min_cls_cnt = min(per_cls_cnt)
        if min_cls_cnt < math.ceil(budget / len(clusters)):
            per_cls_budget = [min_cls_cnt] * len(clusters)
            while sum(per_cls_budget) < budget:
                for cls in range(len(clusters)):
                    if per_cls_budget[cls] < per_cls_cnt[cls]:
                        per_cls_budget[cls] += 1
        else:
            per_cls_budget = [math.ceil(budget / len(clusters)) for _ in per_cls_cnt]

        for _ in range(n_subsets):
            st_time = time.time()
            subset = []
            cluster_idx = 0
            for cluster in cluster_idxs.keys():
                if cluster == 46:
                    print("break")
                idxs = cluster_idxs[cluster]
                cluster_embeddings = embeddings[idxs, :]
                clustered_dist = get_cdist(cluster_embeddings)
                if metric == "rbf_kernel":
                    clustered_sijs = get_rbf_kernel(clustered_dist, kw)
                elif metric == "dot":
                    clustered_sijs = get_dot_product(cluster_embeddings)
                    if submod_function in ["disp_min_pc", "disp_sum_pc"]:
                        clustered_sijs = (clustered_sijs - np.min(clustered_sijs)) / (
                            np.max(clustered_sijs) - np.min(clustered_sijs)
                        )
                    else:
                        if np.min(clustered_sijs) < 0:
                            clustered_sijs = clustered_sijs - np.min(clustered_sijs)
                elif metric == "cossim":
                    normalized_embeddings = cluster_embeddings / np.linalg.norm(
                        cluster_embeddings, axis=1, keepdims=True
                    )
                    clustered_sijs = get_dot_product(normalized_embeddings)
                    if submod_function in ["disp_min_pc", "disp_sum_pc"]:
                        clustered_sijs = (clustered_sijs - np.min(clustered_sijs)) / (
                            np.max(clustered_sijs) - np.min(clustered_sijs)
                        )
                    else:
                        clustered_sijs = (1 + clustered_sijs) / 2
                else:
                    raise ValueError("Please enter a valid metric")

                if submod_function in ["supfl"]:
                    obj = submodlib.FacilityLocationFunction(
                        n=cluster_embeddings.shape[0],
                        separate_rep=False,
                        mode="dense",
                        sijs=clustered_sijs,
                    )
                elif submod_function in ["gc_pc"]:
                    obj = submodlib.GraphCutFunction(
                        n=cluster_embeddings.shape[0],
                        mode="dense",
                        lambdaVal=0.4,
                        separate_rep=False,
                        ggsijs=clustered_sijs,
                    )
                elif submod_function in ["logdet_pc"]:
                    obj = submodlib.LogDeterminantFunction(
                        n=cluster_embeddings.shape[0],
                        mode="dense",
                        lambdaVal=1,
                        sijs=clustered_sijs,
                    )

                elif submod_function == "disp_min_pc":
                    obj = submodlib.DisparityMinFunction(
                        n=cluster_embeddings.shape[0], mode="dense", sijs=clustered_sijs
                    )

                elif submod_function == "disp_sum_pc":
                    obj = submodlib.DisparitySumFunction(
                        n=cluster_embeddings.shape[0], mode="dense", sijs=clustered_sijs
                    )
                # print(budget, per_cls_budget, per_cls_cnt)
                if submod_function in ["disp_min_pc", "gc_pc"]:
                    # print(cluster_idx, per_cls_budget[cluster_idx], cluster_embeddings.shape[0])
                    if per_cls_budget[cluster_idx] == cluster_embeddings.shape[0]:
                        clustergreedyList = obj.maximize(
                            budget=per_cls_budget[cluster_idx] - 1,
                            optimizer="NaiveGreedy",
                            stopIfZeroGain=False,
                            stopIfNegativeGain=False,
                            epsilon=0.1,
                            verbose=False,
                            show_progress=True,
                            costs=None,
                            costSensitiveGreedy=False,
                        )
                        rem_elem = list(
                            set(range(cluster_embeddings.shape[0])).difference(
                                set([x[0] for x in clustergreedyList])
                            )
                        )[0]
                        rem_gain = clustergreedyList[-1][1]
                        clustergreedyList.append((rem_elem, rem_gain))
                    else:
                        clustergreedyList = obj.maximize(
                            budget=per_cls_budget[cluster_idx],
                            optimizer="StochasticGreedy",
                            stopIfZeroGain=False,
                            stopIfNegativeGain=False,
                            epsilon=0.1,
                            verbose=False,
                            show_progress=True,
                            costs=None,
                            costSensitiveGreedy=False,
                        )
                else:
                    # print(cluster_idx, per_cls_budget[cluster_idx], cluster_embeddings.shape[0])
                    if per_cls_budget[cluster_idx] == cluster_embeddings.shape[0]:
                        clustergreedyList = obj.maximize(
                            budget=per_cls_budget[cluster_idx] - 1,
                            optimizer="NaiveGreedy",
                            stopIfZeroGain=False,
                            stopIfNegativeGain=False,
                            epsilon=0.1,
                            verbose=False,
                            show_progress=True,
                            costs=None,
                            costSensitiveGreedy=False,
                        )
                        rem_elem = list(
                            set(range(cluster_embeddings.shape[0])).difference(
                                set([x[0] for x in clustergreedyList])
                            )
                        )[0]
                        rem_gain = clustergreedyList[-1][1]
                        clustergreedyList.append((rem_elem, rem_gain))
                    else:
                        clustergreedyList = obj.maximize(
                            budget=per_cls_budget[cluster_idx],
                            optimizer="LazierThanLazyGreedy",
                            stopIfZeroGain=False,
                            stopIfNegativeGain=False,
                            epsilon=0.1,
                            verbose=False,
                            show_progress=True,
                            costs=None,
                            costSensitiveGreedy=False,
                        )
                cluster_idx += 1
                clusteredgreedylist_with_orig_idxs = [
                    (idxs[x[0]], x[1]) for x in clustergreedyList
                ]
                subset.extend(clusteredgreedylist_with_orig_idxs)
                del obj
            subset.sort(key=lambda x: x[1], reverse=True)
            subsets.append(subset)
            total_time += time.time() - st_time

    print("Average Time for Stochastic Greedy Subset Selection is :", total_time)
    # Sorted Label-wise Partition
    # cluster_idxs = {}
    # greedy_idxs = [x[0] for x in subset]
    # for i in greedy_idxs:
    #     if train_labels[i] in cluster_idxs.keys():
    #         cluster_idxs[train_labels[i]].append(i)
    #     else:
    #         cluster_idxs[train_labels[i]] = [i]
    return subsets


def load_dataset(
    dataset_name, data_dir, seed, return_valid=False, return_test=False, config=None
):
    if dataset_name == "glue_sst2":
        """
        Load GLUE SST2 dataset. We are only using train and validation splits since the test split doesn't come with gold labels. For testing purposes, we use 5% of train
        dataset as test dataset.
        """
        glue_dataset = datasets.load_dataset("glue", "sst2", cache_dir=data_dir)
        fullset = glue_dataset["train"]
        valset = glue_dataset["validation"]
        test_set_fraction = 0.05
        seed = 42
        num_fulltrn = len(fullset)
        num_test = int(num_fulltrn * test_set_fraction)
        num_trn = num_fulltrn - num_test
        trainset, testset = random_split(
            fullset, [num_trn, num_test], generator=torch.Generator().manual_seed(seed)
        )

    elif dataset_name == "trec6":
        trec6_dataset = datasets.load_dataset("trec", cache_dir=data_dir)
        fullset = trec6_dataset["train"]
        testset = trec6_dataset["test"]
        validation_set_fraction = 0.1
        seed = 42
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(
            fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed)
        )

    elif dataset_name == "imdb":
        trec6_dataset = datasets.load_dataset("imdb", cache_dir=data_dir)
        fullset = trec6_dataset["train"]
        testset = trec6_dataset["test"]
        validation_set_fraction = 0.1
        seed = 42
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(
            fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed)
        )

    elif dataset_name == "rotten_tomatoes":
        dataset = datasets.load_dataset("rotten_tomatoes", cache_dir=data_dir)
        trainset = dataset["train"]
        valset = dataset["validation"]
        testset = dataset["test"]

    elif dataset_name == "tweet_eval":
        dataset = datasets.load_dataset("tweet_eval", "emoji", cache_dir=data_dir)
        trainset = dataset["train"]
        valset = dataset["validation"]
        testset = dataset["test"]

    elif dataset_name == "cifar10":
        fullset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=None
        )

        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(
            fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed)
        )

    elif dataset_name == "cifar100":
        fullset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=None
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=None
        )

        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(
            fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed)
        )

    elif dataset_name == "tinyimagenet":
        fullset = TinyImageNet(
            root=data_dir, split="train", download=True, transform=None
        )
        testset = TinyImageNet(
            root=data_dir, split="val", download=True, transform=None
        )
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(
            fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed)
        )

    elif dataset_name == "mnist":
        fullset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=None
        )
        testset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=None
        )
        validation_set_fraction = 0.1
        num_fulltrn = len(fullset)
        num_val = int(num_fulltrn * validation_set_fraction)
        num_trn = num_fulltrn - num_val
        trainset, valset = random_split(
            fullset, [num_trn, num_val], generator=torch.Generator().manual_seed(seed)
        )

    elif dataset_name == "cityscapes":
        crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        trainset = eval("datasets." + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.TRAIN_SET,
            num_samples=None,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=config.TRAIN.MULTI_SCALE,
            flip=config.TRAIN.FLIP,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TRAIN.BASE_SIZE,
            crop_size=crop_size,
            downsample_rate=config.TRAIN.DOWNSAMPLERATE,
            scale_factor=config.TRAIN.SCALE_FACTOR,
        )

        test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
        test_dataset = eval("datasets." + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.TEST_SET,
            num_samples=config.TEST.NUM_SAMPLES,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=False,
            flip=False,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TEST.BASE_SIZE,
            crop_size=test_size,
            downsample_rate=1,
        )

        return_valid = False
    elif dataset_name == "pascal_ctx":
        crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        trainset = eval("datasets." + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.TRAIN_SET,
            num_samples=None,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=config.TRAIN.MULTI_SCALE,
            flip=config.TRAIN.FLIP,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TRAIN.BASE_SIZE,
            crop_size=crop_size,
            downsample_rate=config.TRAIN.DOWNSAMPLERATE,
            scale_factor=config.TRAIN.SCALE_FACTOR,
        )

        test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
        test_dataset = eval("datasets." + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.TEST_SET,
            num_samples=config.TEST.NUM_SAMPLES,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=False,
            flip=False,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TEST.BASE_SIZE,
            crop_size=test_size,
            downsample_rate=1,
        )

        return_valid = False
    else:
        return None

    if not (return_valid and return_test):
        if return_valid:
            return trainset, valset
        elif return_test:
            return trainset, testset
        else:
            return trainset
    else:
        return trainset, valset, testset


def generate_text_similarity_kernel(
    dataset, model, stats=True, seed=42, data_dir="../data", device="cpu"
):
    # Assertion Check:
    assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (
        dataset in list(LABEL_MAPPINGS.keys())
    ), "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."

    # Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    if train_dataset.__class__.__name__ == "Subset":
        train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset]
        train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
    else:
        train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
        train_labels = train_dataset[LABEL_MAPPINGS[dataset]]

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(
        os.path.join(
            os.path.abspath(data_dir), dataset + "_" + model + "_train_embeddings.pkl"
        )
    ):
        train_embeddings = compute_text_embeddings(model, train_sentences, device)
        store_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            ),
            train_embeddings,
        )
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            )
        )

    if not os.path.exists(
        os.path.join(
            os.path.abspath(data_dir), dataset + "_" + model + "_dist_kernel.h5"
        )
    ):
        data_dist = get_cdist(train_embeddings)
        with h5py.File(
            os.path.join(
                os.path.abspath(data_dir), dataset + "_" + model + "_dist_kernel.h5"
            ),
            "w",
        ) as hf:
            hf.create_dataset("dist_kernel", data=data_dist)

        if stats:
            plt.hist(data_dist, bins="auto")
            plt.savefig(dataset + "_" + model + "_dist_hist.png")


def generate_image_similarity_kernel(
    dataset, model, stats=True, seed=42, data_dir="../data", device="cpu"
):
    # Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    train_images = [x[0] for x in train_dataset]
    train_labels = [x[1] for x in train_dataset]

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(
        os.path.join(
            os.path.abspath(data_dir), dataset + "_" + model + "_train_embeddings.pkl"
        )
    ):
        if model[:3] == "ViT":
            train_embeddings = compute_vit_image_embeddings(train_images, device)
        else:
            train_embeddings = compute_image_embeddings(model, train_images, device)
        store_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            ),
            train_embeddings,
        )
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            )
        )

    if not os.path.exists(
        os.path.join(
            os.path.abspath(data_dir), dataset + "_" + model + "_dist_kernel.h5"
        )
    ):
        data_dist = get_cdist(train_embeddings)
        with h5py.File(
            os.path.join(
                os.path.abspath(data_dir), dataset + "_" + model + "_dist_kernel.h5"
            ),
            "w",
        ) as hf:
            hf.create_dataset("dist_kernel", data=data_dist)

        if stats:
            plt.hist(data_dist, bins="auto")
            plt.savefig(dataset + "_" + model + "_dist_hist.png")


def generate_image_global_order(
    dataset,
    model,
    submod_function,
    metric,
    kw,
    r2_coefficient,
    knn,
    seed=42,
    data_dir="../data",
    device="cpu",
    config=None,
    partition_mode="native",
):
    # Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed, config=config)

    if dataset in ["cityscapes", "pascal_ctx"]:
        train_images = []
        train_labels = []
        for x in tqdm(
            train_dataset,
            total=len(train_dataset),
            desc=f"loading {dataset} dataset for global ordering generation",
        ):
            train_images.append(
                Image.fromarray(np.transpose(x[0], (1, 2, 0)), mode="RGB")
            )
            train_labels.append(x[1])
    else:
        train_images = [x[0] for x in train_dataset]
        train_labels = [x[1] for x in train_dataset]

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(
        os.path.join(
            os.path.abspath(data_dir), dataset + "_" + model + "_train_embeddings.pkl"
        )
    ):
        if model == "ViT":
            train_embeddings = compute_vit_image_embeddings(train_images, device)
        elif model == "ViT_cls":
            train_embeddings = compute_vit_cls_image_embeddings(train_images, device)
        elif model == "dino":
            train_embeddings = compute_dino_image_embeddings(train_images, device)
        elif model == "dino_cls":
            train_embeddings = compute_dino_cls_image_embeddings(train_images, device)
        elif model == "oracle":
            train_embeddings = compute_oracle_image_embeddings(train_images, device)
            raise NotImplementedError
        elif model == "sam":
            raise NotImplementedError
        elif model == "segformer":
            train_embeddings = compute_segformer_image_embeddings(train_images, device)
        elif model == "clip":
            train_embeddings = compute_image_embeddings(model, train_images, device)
        else:
            raise NotImplementedError
        store_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            ),
            train_embeddings,
        )
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            )
        )

    # Load global order from pickle file if it exists otherwise compute them and store them.
    if not (
        os.path.exists(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_global_order.pkl",
            )
        )
        and os.path.exists(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(r2_coefficient)
                + "_global_r2.pkl",
            )
        )
        and os.path.exists(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(knn)
                + "_global_knn.pkl",
            )
        )
    ):
        global_order, global_knn, global_r2, cluster_idxs = compute_global_ordering(
            train_embeddings,
            submod_function=submod_function,
            kw=kw,
            r2_coefficient=r2_coefficient,
            knn=knn,
            train_labels=train_labels,
            metric=metric,
            partition_mode=partition_mode,
            train_dataset=train_dataset,
        )
        dict2pickle(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_global_order.pkl",
            ),
            {"globalorder": global_order, "cluster_idxs": cluster_idxs},
        )
        dict2pickle(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(r2_coefficient)
                + "_global_r2.pkl",
            ),
            {"globalr2": global_r2, "cluster_idxs": cluster_idxs},
        )
        dict2pickle(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(knn)
                + "_global_knn.pkl",
            ),
            {"globalknn": global_knn, "cluster_idxs": cluster_idxs},
        )
    else:
        global_order = pickle2dict(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_global_order.pkl",
            ),
            "globalorder",
        )
        cluster_idxs = pickle2dict(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_global_order.pkl",
            ),
            "cluster_idxs",
        )
        global_r2 = pickle2dict(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(r2_coefficient)
                + "_global_r2.pkl",
            ),
            "globalr2",
        )
        global_knn = pickle2dict(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(knn)
                + "_global_knn.pkl",
            ),
            "globalknn",
        )
    return global_order, global_knn, global_r2, cluster_idxs


def generate_text_global_order(
    dataset,
    model,
    submod_function,
    metric,
    kw,
    r2_coefficient,
    knn,
    seed=42,
    data_dir="../data",
    device="cpu",
):
    # Assertion Check:
    assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (
        dataset in list(LABEL_MAPPINGS.keys())
    ), "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."

    # Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    if train_dataset.__class__.__name__ == "Subset":
        train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset]
        train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
    else:
        train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
        train_labels = train_dataset[LABEL_MAPPINGS[dataset]]

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(
        os.path.join(
            os.path.abspath(data_dir), dataset + "_" + model + "_train_embeddings.pkl"
        )
    ):
        train_embeddings = compute_text_embeddings(model, train_sentences, device)
        store_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            ),
            train_embeddings,
        )
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            )
        )

    # Load global order from pickle file if it exists otherwise compute them and store them.
    if not (
        os.path.exists(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_global_order.pkl",
            )
        )
        and os.path.exists(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(r2_coefficient)
                + "_global_r2.pkl",
            )
        )
        and os.path.exists(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(knn)
                + "_global_knn.pkl",
            )
        )
    ):
        global_order, global_knn, global_r2, cluster_idxs = compute_global_ordering(
            train_embeddings,
            submod_function=submod_function,
            kw=kw,
            r2_coefficient=r2_coefficient,
            knn=knn,
            train_labels=train_labels,
            metric=metric,
        )
        dict2pickle(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_global_order.pkl",
            ),
            {"globalorder": global_order, "cluster_idxs": cluster_idxs},
        )
        dict2pickle(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(r2_coefficient)
                + "_global_r2.pkl",
            ),
            {"globalr2": global_r2, "cluster_idxs": cluster_idxs},
        )
        dict2pickle(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(knn)
                + "_global_knn.pkl",
            ),
            {"globalknn": global_knn, "cluster_idxs": cluster_idxs},
        )
    else:
        global_order = pickle2dict(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_global_order.pkl",
            ),
            "globalorder",
        )
        cluster_idxs = pickle2dict(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_global_order.pkl",
            ),
            "cluster_idxs",
        )
        global_r2 = pickle2dict(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(r2_coefficient)
                + "_global_r2.pkl",
            ),
            "globalr2",
        )
        global_knn = pickle2dict(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(knn)
                + "_global_knn.pkl",
            ),
            "globalknn",
        )
    return global_order, global_knn, global_r2, cluster_idxs


def generate_image_stochastic_subsets(
    dataset,
    model,
    submod_function,
    metric,
    kw,
    fraction,
    n_subsets,
    stochastic_subsets_file,
    seed=42,
    data_dir="../data",
    device="cpu",
    config=None,
    partition_mode="native",
    epsilon=0.001,
):
    # Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed, config=config)

    if dataset in ["cityscapes", "pascal_ctx"]:
        train_images = []
        train_labels = []
        # FIXME: This takes an inordinately long time
        for x in tqdm(
            train_dataset,
            total=len(train_dataset),
            desc=f"loading {dataset} dataset for stochastic subset generation",
        ):
            train_images.append(
                Image.fromarray(np.transpose(x[0], (1, 2, 0)), mode="RGB")
            )
            train_labels.append(x[1])
    else:
        train_images = [x[0] for x in train_dataset]
        train_labels = [x[1] for x in train_dataset]

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(
        os.path.join(
            os.path.abspath(data_dir), dataset + "_" + model + "_train_embeddings.pkl"
        )
    ):
        if model == "ViT":
            train_embeddings = compute_vit_image_embeddings(train_images, device)
        elif model == "ViT_cls":
            train_embeddings = compute_vit_cls_image_embeddings(train_images, device)
        elif model == "dino":
            train_embeddings = compute_dino_image_embeddings(train_images, device)
        elif model == "dino_cls":
            train_embeddings = compute_dino_cls_image_embeddings(train_images, device)
        elif model == "oracle_spat":
            train_embeddings = compute_oracle_image_embeddings(
                train_images, device, mode="oracle_spat"
            )
        elif model == "oracle_context":
            train_embeddings = compute_oracle_image_embeddings(
                train_images, device, mode="oracle_context"
            )
        elif model == "oracle_feature_flat":
            train_embeddings = compute_oracle_image_embeddings(
                train_images, device, mode="oracle_feature_flat"
            )
        elif model == "sam":
            raise NotImplementedError
        elif model == "segformer":
            train_embeddings = compute_segformer_image_embeddings(train_images, device)
        elif model == "clip":
            train_embeddings = compute_image_embeddings(model, train_images, device)
        else:
            raise NotImplementedError
        store_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            ),
            train_embeddings,
        )
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            )
        )

    # Load stochastic subsets from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(stochastic_subsets_file):
        stochastic_subsets = compute_stochastic_greedy_subsets(
            train_embeddings,
            submod_function,
            train_labels,
            kw,
            metric,
            fraction,
            n_subsets=n_subsets,
            partition_mode=partition_mode,
            train_dataset=train_dataset,  # FIXME: Drunk hack
            epsilon=epsilon,
        )
        dict2pickle(
            stochastic_subsets_file,
            {"stochastic_subsets": stochastic_subsets},
        )
    else:
        stochastic_subsets = pickle2dict(
            stochastic_subsets_file,
            "stochastic_subsets",
        )
    return stochastic_subsets


def generate_text_stochastic_subsets(
    dataset,
    model,
    submod_function,
    metric,
    kw,
    fraction,
    n_subsets,
    seed=42,
    data_dir="../data",
    device="cpu",
):
    # Assertion Check:
    assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (
        dataset in list(LABEL_MAPPINGS.keys())
    ), "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."

    # Load Dataset
    train_dataset = load_dataset(dataset, data_dir, seed)

    if train_dataset.__class__.__name__ == "Subset":
        train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset]
        train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
    else:
        train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
        train_labels = train_dataset[LABEL_MAPPINGS[dataset]]

    # Load embeddings from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(
        os.path.join(
            os.path.abspath(data_dir), dataset + "_" + model + "_train_embeddings.pkl"
        )
    ):
        train_embeddings = compute_text_embeddings(model, train_sentences, device)
        store_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            ),
            train_embeddings,
        )
    else:
        # Load the embeddings from disc
        train_embeddings = load_embeddings(
            os.path.join(
                os.path.abspath(data_dir),
                dataset + "_" + model + "_train_embeddings.pkl",
            )
        )

    # Load stochastic subsets from pickle file if it exists otherwise compute them and store them.
    if not os.path.exists(
        os.path.join(
            os.path.abspath(data_dir),
            dataset
            + "_"
            + model
            + "_"
            + metric
            + "_"
            + submod_function
            + "_"
            + str(kw)
            + "_"
            + str(fraction)
            + "_stochastic_subsets.pkl",
        )
    ):
        stochastic_subsets = compute_stochastic_greedy_subsets(
            train_embeddings,
            submod_function,
            train_labels,
            kw,
            metric,
            fraction,
            n_subsets=n_subsets,
        )
        dict2pickle(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_"
                + str(fraction)
                + "_stochastic_subsets.pkl",
            ),
            {"stochastic_subsets": stochastic_subsets},
        )
    else:
        stochastic_subsets = pickle2dict(
            os.path.join(
                os.path.abspath(data_dir),
                dataset
                + "_"
                + model
                + "_"
                + metric
                + "_"
                + submod_function
                + "_"
                + str(kw)
                + "_"
                + str(fraction)
                + "_stochastic_subsets.pkl",
            ),
            "stochastic_subsets",
        )
    return stochastic_subsets


# def analyze_go_wt_diff_init(dataset, model, submod_function, data_dir='../data', device='cpu', seed=42):

#     #Load Arguments
#     #args = parse_args()

#     #Assertion Check:
#     assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (dataset in list(LABEL_MAPPINGS.keys())), \
#     "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."

#     #Load Dataset
#     train_dataset = load_dataset(dataset, data_dir, seed)

#     if train_dataset.__class__.__name__ == 'Subset':
#         train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset]
#         train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
#     else:
#         train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
#         train_labels = train_dataset[LABEL_MAPPINGS[dataset]]

#     # Load embeddings from pickle file if it exists otherwise compute them and store them.
#     if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_train_embeddings.pkl')):
#         train_embeddings = compute_text_embeddings(model, train_sentences, device)
#         store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_train_embeddings.pkl'), train_embeddings)
#     else:
#         # Load the embeddings from disc
#         train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_train_embeddings.pkl'))

#     groundset = list(range(train_embeddings.shape[0]))
#     random_inits = random.sample(groundset, 10)
#     subsets = []

#     for random_init in random_inits:
#         remset = [x for x in groundset if x != random_init]
#         remdata = train_embeddings[remset]
#         privatedata = train_embeddings[random_init].reshape(1, -1)
#         global_order = compute_global_ordering(remdata, submod_function= submod_function, train_labels=train_labels, private_embeddings=privatedata)
#         subset = [random_init]
#         rem_subset = [remset[x[0]] for x in global_order]
#         subset.extend(rem_subset)
#         subsets.append(subset)

#     budget = int(0.3 * train_embeddings.shape[0])

#     common_fraction = np.zeros((len(subsets), len(subsets)))
#     for i in range(len(subsets)):
#         for j in range(len(subsets)):
#            common_fraction[i][j] = len(set(subsets[i][:budget]).intersection(set(subsets[j][:budget])))/len(set(subsets[i][:budget]))
#     return common_fraction


# def analyze_go_label_dists(dataset, model, submod_function, data_dir='../data', device='cpu', seed=42):

#     #Load Arguments
#     #args = parse_args()
#     fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
#     label_dists_dict = {}
#     #Assertion Check:
#     assert (dataset in list(SENTENCE_MAPPINGS.keys())) and (dataset in list(LABEL_MAPPINGS.keys())), \
#     "Please add the SENTENCE and LABEL column names to the SENTENCE_MAPPING and LABEL_MAPPINGS dictionaries in generate_global_order.py file."

#     #Load Dataset
#     train_dataset = load_dataset(dataset, data_dir, seed)

#     if train_dataset.__class__.__name__ == 'Subset':
#         train_sentences = [x[SENTENCE_MAPPINGS[dataset]] for x in train_dataset]
#         train_labels = [x[LABEL_MAPPINGS[dataset]] for x in train_dataset]
#     else:
#         train_sentences = train_dataset[SENTENCE_MAPPINGS[dataset]]
#         train_labels = train_dataset[LABEL_MAPPINGS[dataset]]

#     # Load embeddings from pickle file if it exists otherwise compute them and store them.
#     if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_train_embeddings.pkl')):
#         train_embeddings = compute_text_embeddings(model, train_sentences, device)
#         store_embeddings(os.path.join(os.path.abspath(data_dir), dataset  + '_train_embeddings.pkl'), train_embeddings)
#     else:
#         # Load the embeddings from disc
#         train_embeddings = load_embeddings(os.path.join(os.path.abspath(data_dir), dataset + '_train_embeddings.pkl'))

#     if not os.path.exists(os.path.join(os.path.abspath(data_dir), dataset + '_' + submod_function + '_global_order.pkl')):
#         global_order, global_knn, global_r2 = compute_global_ordering(train_embeddings, submod_function= submod_function, train_labels=train_labels)
#         store_globalorder(os.path.join(os.path.abspath(data_dir), dataset + '_' + submod_function + '_global_order.pkl'), global_order, global_knn, global_r2)
#     else:
#         global_order, global_knn, global_r2 = load_globalorder(os.path.join(os.path.abspath(data_dir), dataset + '_' + submod_function + '_global_order.pkl'))

#     global_order_idxs = [x[0] for x in global_order]
#     num_labels = len(set(train_labels))

#     for fraction in fractions:
#         budget = int(fraction * train_embeddings.shape[0])
#         label_dist = np.array([0]*num_labels)
#         for i in range(budget):
#             label = train_labels[global_order_idxs[i]]
#             label_dist[label] += 1
#         label_dist = label_dist/budget
#         label_dists_dict[fraction] = label_dist
#     return label_dists_dict


if __name__ == "__main__":
    # args = parse_args()
    # submod_functions = ['fl', 'supfl', 'gc', 'logdet']
    # label_dists_dict = analyze_go_label_dists(args.dataset, args.model, args.submod_function, data_dir=args.data_dir, device=args.device, seed=args.seed)
    train_dataset = load_dataset("cifar10", "../data", 42)

    train_images = [x[0] for x in train_dataset]
    train_labels = [x[1] for x in train_dataset]
    compute_vit_image_embeddings(train_images, "cuda")

    print()
