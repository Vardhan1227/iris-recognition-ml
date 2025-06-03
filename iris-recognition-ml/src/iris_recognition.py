import os
from preprocessing import load_and_preprocess_images
from feature_extraction import extract_hog_features
from model import train_and_evaluate_model

if __name__ == "__main__":
    data_path = "dataset"
    images, labels = load_and_preprocess_images(data_path)
    features = extract_hog_features(images)
    train_and_evaluate_model(features, labels)
