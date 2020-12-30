import os
import random
import numpy as np
import dask.dataframe as dd
from sklearn.decomposition import PCA
from skimage import io
from skimage.transform import resize
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray
import pickle as pk

import warnings
warnings.filterwarnings("ignore")

def get_hog(image):
    hog_ft,_ = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=False)
    return hog_ft

def get_lbp(image):
    method = 'nri_uniform'
    radius = 3
    n_points = 8

    lbp = local_binary_pattern(image, n_points, radius, method)

    #histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    lbp_59 = np.zeros(59)
    lbp_59[:hist.shape[0],]=hist
    return lbp_59

def pca_models():
    files = build_image_set()
    hog_features = list()
    lbp_features = list()
    print('extracting features')
    for pic in files:
        img = io.imread(pic,pilmode="RGB")
        img1 = resize(img,(227,227,3))
        img2 = rgb2gray(img1)

        hog_ft = get_hog(img2)
        lbp_ft = get_lbp(img2)
        
        hog_features.append(hog_ft)
        lbp_features.append(lbp_ft)

    del files
    hog_matrix = np.array(hog_features)
    lbp_matrix = np.array(lbp_features)
    np.save('/home/ubuntu/imsim/local_global_features/features_for_pca/hog_matrix.npy',hog_matrix)
    np.save('/home/ubuntu/imsim/local_global_features/features_for_pca/lbp_matrix.npy',lbp_matrix)
    print('HOG and LBP features saved to .npy files in the features_for_pca folder')
    print('building PCA on hog')
    #instantiate PCA to reduce hog features from 26244 to 59
    hog_pca = PCA(n_components = 59,whiten=True)
    hog_components = hog_pca.fit_transform(hog_matrix)
    explained_variance = hog_pca.explained_variance_ratio_
    #Use h5p or spark
    pk.dump(hog_pca, open("hog_pca.pkl","wb"))
    print('saved hog_pca model')
    print('hog pca 59 explained variance',np.sum(explained_variance))

    #combine hog and lbp
    handcrafted = np.concatenate((hog_components,lbp_matrix),axis=1)
    del hog_components, hog_matrix
    #instantiate handcrafted PCA
    print('pca on handcrafted features')
    handcrafted_pca = PCA(n_components=64,whiten=True)
    handcrafted_pca.fit(handcrafted)
    explained_variance1 = handcrafted_pca.explained_variance_ratio_
    #Use h5p or spark
    pk.dump(handcrafted_pca, open("handcrafted_pca_154.pkl","wb"))
    print('saved final handcrafted pca model')
    print('final pca 64 explained variance',np.sum(explained_variance1))

def build_image_set():
    
    images = []
    photo_loc = r'/home/ubuntu/imsim/pixsy_verified_dataset/photos'
    match_loc = r'/home/ubuntu/imsim/pixsy_verified_dataset/matches'
    photos = list(set(os.listdir(photo_loc)))
    matches = list(set(os.listdir(match_loc)))
    p_count = m_count = 7500
    p_random = random.choices(photos,k=15000)
    m_random = random.choices(matches,k=55000)
    print('started building photo list')
    c=0
    for p in p_random:
        try:
            pic = os.path.join(photo_loc,p)
            io.imread(pic,pilmode="RGB")
            images.append(pic)
            c+=1
            if c % 500 == 0:
                print('done with a set of 500 images')
            if c == p_count:
                break
        except: continue
    print('done building photo list')
    print('started building match list')
    i=0
    for m in m_random:
        try:
            pic = os.path.join(match_loc,m)
            io.imread(pic,pilmode="RGB")
            images.append(pic)
            i+=1
            if i % 500 == 0:
                print('done with a set of 500 images')
            if i == m_count:
                break
        except: continue
    print('done building match list')
    return images

pca_models()