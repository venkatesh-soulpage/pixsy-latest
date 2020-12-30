import os
import time
import json
import warnings
import argparse
import pickle as pk
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray


import torch
from torchvision import transforms
import sys

sys.path.insert(0, "ml/networks")

import deep_rank_net

# from ml.networks import deep_rank_net


# from kafkaservice import kafkaproducer

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    model = torch.load(
        os.path.join(
            "ml", "models", "deepranknet_pixsy_data_unfreeze_avg_maxpool_64.model"
        ),
        map_location=device,
    )
else:
    model = torch.load(
        os.path.join(
            "ml", "models", "deepranknet_pixsy_data_unfreeze_avg_maxpool_64.model"
        )
    )
print("deep model loaded")
model.eval()


def get_hog(image):
    """
    DOCSTRINGS
    """

    hog_ft, _ = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        multichannel=False,
    )
    return hog_ft


def get_lbp(image):
    method = "nri_uniform"
    radius = 3
    n_points = 8

    lbp = local_binary_pattern(image, n_points, radius, method)

    # histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    lbp_59 = np.zeros(59)
    lbp_59[
        : hist.shape[0],
    ] = hist
    return lbp_59


def get_deep(image):

    img = torch.FloatTensor(image).permute(2, 0, 1)

    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    inp = torch.unsqueeze(transform(img), 0).to(device)
    deep_ft = model(inp).cpu().detach().numpy().ravel()
    return deep_ft


def extract_features(files):

    idx = list()
    hog_features = list()
    lbp_features = list()
    deep_features = list()
    start = time.time()
    for pic in files:
        img = io.imread(pic, pilmode="RGB")

        img1 = resize(img, (227, 227, 3))
        img2 = rgb2gray(img1)

        hog_ft = get_hog(img2)
        lbp_ft = get_lbp(img2)
        deep_ft = get_deep(img1)

        hog_features.append(hog_ft)
        lbp_features.append(lbp_ft)
        deep_features.append(deep_ft)
        idx.append(pic)  # using the image filename as the index for the feature

    hog_matrix = np.array(hog_features)
    lbp_matrix = np.array(lbp_features)
    deep_matrix = np.array(deep_features)

    # Load pre-created PCA model to reduce hog features from 26244 to 59
    with open("models/hog_pca.pkl", "rb") as f:
        hog_pca = pk.load(f)
    hog_components = hog_pca.transform(hog_matrix)

    # combine hog and lbp
    handcrafted = np.concatenate((hog_components, lbp_matrix), axis=1)

    # load handcrafted PCA
    with open("models/handcrafted_pca.pkl", "rb") as g:
        handcrafted_pca = pk.load(g)
    handcrafted_components = handcrafted_pca.transform(handcrafted)
    combined_features = np.concatenate((handcrafted_components, deep_matrix), axis=1)
    end = time.time()
    df = pd.DataFrame(combined_features, index=idx).drop_duplicates()

    df.index.rename("name", inplace=True)
    data = json.loads(df.to_json(orient="table"))
    data = data["data"]
    names = list(map(lambda d: d.pop("name"), data))
    producer_data = []
    for index in range(len(names)):
        producer_data.append({"name": names[index], "data": data[index]})
    # kafkaproducer(producer_data, im_type)

    # if im_type == "photos":
    #     index_name = "photo_id"
    # elif im_type == "matches":
    #     index_name = "match_id"
    # df.index.rename(index_name, inplace=True)

    print(
        "time taken extract features for {} images is {} seconds".format(
            df.shape[0], end - start
        )
    )
