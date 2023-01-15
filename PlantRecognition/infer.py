import config
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
import cv2
import encoder_model
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_image_tensor(image_path, device):
    """
    Load a given image to device
    :param image_path: Path to image to be loaded
    :param device: cuda or cpu
    :return:
    """
    image_tensor = Image.open(image_path)
    transforms = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    image_tensor = transforms(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def compute_similar_images(encoder_model1, image_path, num_images, embedding, device):
    """
    Given an image and a number of similar images to generate
    :param encoder_model1:
    :param image_path: Path to image whose similar images are to be found.
    :param num_images: Number of similar images to find.
    :param embedding: A (num_images, embedding_dim) Embedding of images learnt from auto-encoder
    :param device:cuda or cpu
    :return:  Returns the num_images closest nearest images.
    """
    image_tensor = load_image_tensor(image_path, device)
    with torch.no_grad():
        image_embedding = encoder_model1(image_tensor).cpu().detach().numpy()

    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    knn = NearestNeighbors(n_neighbors=num_images, metric="jaccard")
    knn.fit(embedding)
    _, indices = knn.kneighbors(flattened_embedding)
    indices_list1 = indices.tolist()
    return indices_list1


def plot_similar_images(indices_list1):
    """
    Plot images that are similar to indices obtained from computing similar images
    :param indices_list1: List of List of indexes
    :return:
    """

    indices = indices_list1[0]
    for index in indices:
        if index == 0:
            pass
        else:
            img_name = str(index - 1) + ".jpg"
            img_path = os.path.join(config.DATA_PATH + img_name)
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.show()
            img.save(f"outputs/image_1/recommended_{index - 1}.jpg")


def compute_similar_features(image_path, num_images, embedding, nfeatures=30):
    """
    Given an image, it computes features using ORB detector and finds similar images to it
    :param image_path:
    :param num_images:
    :param embedding:
    :param nfeatures:
    :return:
    """

    image = cv2.imread(image_path)
    orb = cv2.ORB_create(nfeatures=nfeatures)

    keypoint_features = orb.detect(image)
    # compute the descriptors with ORB

    keypoint_features, des = orb.compute(image, keypoint_features)

    des = des / 255.0
    des = np.expand_dims(des, axis=0)
    des = np.reshape(des, (des.shape[0], -1))
    # print(des.shape)
    # print(embedding.shape)
    # print(des.shape[-1])
    pca = PCA(n_components=des.shape[-1])
    reduced_embedding = pca.fit_transform(
        embedding,
    )
    # print(reduced_embedding.shape)

    knn = NearestNeighbors(n_neighbors=num_images, metric="jaccard")
    knn.fit(reduced_embedding)
    _, indices = knn.kneighbors(des)

    indices_list1 = indices.tolist()
    # print(indices_list)
    return indices_list1


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder = encoder_model.ConvEncoder()

    # Load the state dict of encoder
    encoder.load_state_dict(torch.load(config.ENCODER_MODEL_PATH, map_location=device))
    encoder.eval()
    encoder.to(device)

    # Loads the embedding
    embedding = np.load(config.EMBEDDING_PATH)

    indices_list = compute_similar_images(
        encoder, config.TEST_IMAGE_PATH, config.NUM_IMAGES, embedding, device
    )
    img_path = os.path.join(config.TEST_IMAGE_PATH)
    img = Image.open(img_path).convert("RGB")
    plt.imshow(img)
    plt.show()
    plot_similar_images(indices_list)
    indices_list1 = compute_similar_features(config.TEST_IMAGE_PATH, 5, embedding)
    plot_similar_images(indices_list1)


