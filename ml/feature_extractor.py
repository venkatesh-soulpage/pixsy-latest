import os
import time
import uuid
import torch
import imutils
import warnings
import numpy as np
import pandas as pd
import pickle as pk
from PIL import Image

from cv2 import cv2
from skimage import io
from skimage.transform import resize
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms


from ml.networks import DeepRank
from ml.networks import ScaleTripletModel as ScaleNet
from server.kafkaservice import kafkaproducer

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

scalenet_model = ScaleNet()
deeprank_model = DeepRank()

# loading the deep learning trained model weights into the model objects created above
if device == "cpu":

    scalenet_wts = torch.load(
        os.path.join("ml", "models", "allcnn_2x512_dual_pool_2bn.pt"),
        map_location=device,
    )
    scalenet_model.load_state_dict(scalenet_wts)

    deeprank_wts = torch.load(
        os.path.join("ml", "models", "resnet18_512.pt"),
        map_location=device,
    )
    deeprank_model.load_state_dict(deeprank_wts)

else:
    scalenet_wts = torch.load(
        os.path.join("ml", "models", "allcnn_2x512_dual_pool_2bn.pt"),
    )
    scalenet_model.load_state_dict(scalenet_wts)

    deeprank_wts = torch.load(
        os.path.join("ml", "models", "resnet18_512.pt"),
    )
    deeprank_model.load_state_dict(deeprank_wts)

scalenet_model.eval()
deeprank_model.eval()
print("models loaded")


def get_lbp(image, n_points=8, radius=3, n_bins=256):
    """
    extracts local binary pattern features of an image
    uses the 'skimage' libary's local_binary_pattern method
    arguments: image (image for feature extraction)
    returns a 256 sized vector with LBP features
    """

    method = "default"

    lbp = local_binary_pattern(image, n_points, radius, method)
    # histogram
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    lbp_ft = np.zeros(n_bins)
    lbp_ft[
        : hist.shape[0],
    ] = hist

    return lbp_ft


def get_hog(image):
    """
    extracts histogram of oriented gradient features of an image
    arguments: image (image for feature extraction)
    uses the 'skimage' libary's hog method
    returns the HOG feature vector
    """

    hog_ft = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        multichannel=False,
    )

    return hog_ft


def min_64_resize(G):
    """
    Resizes an image to 64 on the small side retaining aspect ration.
    If height is less than 64, height is changed to 64 if width is less than 64, width is changed
    This is done to avoid the error caused when forward passing small sized images through scalenet model
    """
    shape = G.shape[:2]
    if shape.index(min(shape)) == 0:
        height = 64
        G = imutils.resize(G, height=height)
    else:
        width = 64
        G = imutils.resize(G, width=width)
    return G


def get_deep(image):
    """
    Extracts deep learning model featurs. Both resnet model trained using triplet loss for global feature extraction and
        scalenet model for scale invariant features.
    input: takes an image
    returns resnet feature vector (size 512) and scale model feature vector (size 1024)
    """

    # res18 triplet global features extraction
    img1 = resize(image, (227, 227, 3))
    img1 = torch.FloatTensor(img1).permute(2, 0, 1)
    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    inp = torch.unsqueeze(transform(img1), 0).to(device)
    deeprank_ft = deeprank_model.forward_one(inp).cpu().detach().numpy().ravel()

    # scalenet pyramid pooled features
    transform_pyr = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    G = image.copy()

    if min(G.shape[:2]) < 64:
        G = min_64_resize(G)
    imG = Image.fromarray(G)
    imG = transform_pyr(imG)

    im_pyramid = [imG]
    # creating image pyramid
    c = 0
    while c < 5:
        G = cv2.pyrDown(G)
        shape = G.shape[:2]
        if min(shape) < 64:
            G = min_64_resize(G)
            imG = Image.fromarray(G)
            imG = transform_pyr(imG)
            im_pyramid.append(imG)
            break
        else:
            imG = Image.fromarray(G)
            imG = transform_pyr(imG)
            im_pyramid.append(imG)
        c += 1

    # extracting features of each image in the pyramid and average pooling them
    scale_ft = []
    for scaled_candidate in im_pyramid:
        scaled_candidate = scaled_candidate.unsqueeze(0).to(device)
        try:
            sc_ft = scalenet_model.forward_one(scaled_candidate)
        except:
            sc_ft = torch.zeros((1, 1024))
        scale_ft.append(sc_ft)

    if len(scale_ft) > 1:
        scale_feat = torch.stack(scale_ft)
        scale_feat = torch.mean(scale_feat, dim=0)  # average pooling the features
    else:
        scale_feat = scale_ft[0]

    scalenet_ft = scale_feat.cpu().detach().numpy().ravel()

    return deeprank_ft, scalenet_ft


def rename_columns(df, feat_type):
    """
    utility function used to rename columns of the test dataset
    to match exactly the names used while training the model.
    This is required to predict using the pre-trained classification model
    """
    old_names = list(df.columns)
    new_name_dict = dict(zip(old_names, [str(i) + "_" + feat_type for i in old_names]))
    df = df.rename(columns=new_name_dict)
    return df


def extract_features(files, im_type):
    """
    Feature extraction phase.
    input: image type. Either photos or matches
    returns a dataframe with 2048 features for each image
    """

    idx = list()
    hog_features = list()
    lbp_features = list()
    deep_features = list()
    scale_features = list()

    for f in files:
        # pic = os.path.join(folder_name, f)
        try:
            img = io.imread(f.file, pilmode="RGB")
        except:
            continue

        img1 = resize(img, (227, 227, 3))
        img2 = rgb2gray(img1)

        hog_ft = get_hog(img2)
        lbp_ft = get_lbp(img2)

        deep_ft, scale_ft = get_deep(img)

        hog_features.append(hog_ft)
        lbp_features.append(lbp_ft)
        deep_features.append(deep_ft)
        scale_features.append(scale_ft)

        idx.append(f.filename)  # using the image filename as the index for the feature

    hog_matrix = np.array(hog_features)
    lbp_matrix = np.array(lbp_features)
    deep_matrix = np.array(deep_features)
    scale_matrix = np.array(scale_features)

    # Load pre-created PCA model to reduce hog features to 512
    with open("ml/models/hog_pca_512.pkl", "rb") as f:
        hog_pca = pk.load(f)
    hog_components = hog_pca.transform(hog_matrix)
    # combine hog and lbp
    handcrafted = np.concatenate((hog_components, lbp_matrix), axis=1)
    # load handcrafted PCA
    with open("ml/models/handcrafted_pca_512.pkl", "rb") as g:
        handcrafted_pca = pk.load(g)
    # reducing combined handcrafted feature vector size to 512
    handcrafted_components = handcrafted_pca.transform(handcrafted)
    handcrafted_df = pd.DataFrame(handcrafted_components)
    handcrafted_df = rename_columns(handcrafted_df, "hand")

    deep_df = pd.DataFrame(deep_matrix)
    deep_df = rename_columns(deep_df, "deep")
    scale_df = pd.DataFrame(scale_matrix)
    scale_df = rename_columns(scale_df, "scale")

    df = pd.concat(
        (handcrafted_df, deep_df, scale_df), axis="columns"
    ).drop_duplicates()

    if im_type == "photos":
        df["photo_id"] = idx
    elif im_type == "matches":
        df["match_id"] = idx

    message_id = str(uuid.uuid4())
    kafkaproducer(df.to_json(), im_type, message_id)
    return message_id
