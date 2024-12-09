import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Importing tqdm for progress bar
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load images from the specified directories and assign labels
def load_images_and_labels(image_dirs, label_names):
    paths = []
    labels = []
    for label_idx, label in enumerate(label_names):
        image_dir = os.path.join(image_dirs, label)
        images = os.listdir(image_dir)
        for img in tqdm(images, desc=f"Loading {label}"):
            img_path = os.path.join(image_dir, img)
            paths.append(img_path)
            labels.append(label_idx)  
    return paths, labels

# Extract HOG features Function
def extract_hog_features(image_path):
    try:
        # Read and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features from the grayscale image
        fd, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        
        return fd 
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function to extract features in parallel
def extract_features_parallel(image_paths):
    features = []
    for image_path in tqdm(image_paths, desc="Extracting Features", total=len(image_paths)):
        feature = extract_hog_features(image_path)
        if feature is not None:
            features.append(feature)
    
    return features

def prepare_data(image_dirs):
    label_names = ["HGSC", "OTHERS"]  # Define class names
    paths, labels = load_images_and_labels(image_dirs, label_names)
    
    # Split data into training and testing sets (80% train, 20% test)
    train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.2, random_state=42)
    
    return train_paths, test_paths, train_labels, test_labels


def main():
    start_time = time.time()
    
    image_dirs = "/Users/ghrodri17/Downloads/SVM-DATA" #Use Your own path
    train_paths, test_paths, train_labels, test_labels = prepare_data(image_dirs)
    
    # Extract features from the training set
    train_features = extract_features_parallel(train_paths)
    
    # Convert features and labels to numpy arrays
    train_features = np.array([f for f in train_features if f is not None])
    train_labels = np.array([train_labels[i] for i in range(len(train_features)) if train_features[i] is not None])
    
    # Perform dimensionality reduction (PCA)
    pca = PCA(n_components=90)  # Specify number of components for PCA
    X_train_pca = pca.fit_transform(train_features)
    
    # Train an SVM classifier with hyperparameter tuning using Grid Search
    svc = SVC()
    param_grid = {
        'C': [0.001],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2]
    }
    grid_search = GridSearchCV(svc, param_grid, cv=4, verbose=3, n_jobs=4)
    grid_search.fit(X_train_pca, train_labels)
    
    # Extract features for the test data and transform them using PCA
    test_features = extract_features_parallel(test_paths)
    test_features = np.array([f for f in test_features if f is not None])
    test_labels = np.array([test_labels[i] for i in range(len(test_features)) if test_features[i] is not None])
    X_test_pca = pca.transform(test_features)
    
    # Evaluate the model performance 
    y_pred = grid_search.predict(X_test_pca)
    
    # Print classification report and confusion matrix
    print(classification_report(test_labels, y_pred))
    cm = confusion_matrix(test_labels, y_pred)
    print(cm)
    
    #  Visualize and save the confusion matrix as a plot
    label_names = ["HGSC", "OTHERS"]  # Define the class labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png")
    
    
    print("Total execution time:", time.time() - start_time, "seconds.")

if __name__ == "__main__":
    main()

