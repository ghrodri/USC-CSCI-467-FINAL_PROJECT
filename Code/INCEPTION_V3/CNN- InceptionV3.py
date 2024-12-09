import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_recall_curve

# Directory setup
base_dir = '/Users/ghrodri17/Downloads/SVM-DATA'
IMG_HEIGHT, IMG_WIDTH = 299, 299  # InceptionV3 expects 299x299 images
BATCH_SIZE = 32

# Load image paths and labels
data = []

# Loop through 'HGSC' and 'OTHERS' folders
for class_name in ['HGSC', 'OTHERS']:
    class_dir = os.path.join(base_dir, class_name)
    
    if os.path.exists(class_dir):
        for filename in os.listdir(class_dir):
            if filename.endswith('.png'):
                filepath = os.path.join(class_dir, filename)
                data.append([filepath, class_name])

# Create DataFrame
df = pd.DataFrame(data, columns=['filepath', 'label'])

# Split features (X) and labels (y)
X = df['filepath']
y = df['label']

# Encode labels (Negative=0, Positive=1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train (70%), validation (10%), and test (20%)
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_temp, y_train_temp, test_size=0.10, random_state=42)

# Function to preprocess images
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # InceptionV3 expects 299x299 images
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.applications.inception_v3.preprocess_input(img)  # InceptionV3 preprocessing
    return img

# Image generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    shear_range=0.2,  # Shear transformation
    brightness_range=[0.8, 1.2],  # Adjust brightness
    channel_shift_range=20.0  # Adjust color saturation
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df.iloc[X_train.index],  # Select training file paths
    x_col='filepath',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Binary classification
    shuffle=True
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=df.iloc[X_validation.index],  # Select validation file paths
    x_col='filepath',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Do not shuffle data for evaluation
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df.iloc[X_test.index],  # Select test file paths
    x_col='filepath',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Do not shuffle test data
)

# Load InceptionV3 pre-trained model without top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Unfreeze some top layers for fine-tuning
base_model.trainable = True
fine_tune_at = 249  # Unfreeze the last 30 layers of InceptionV3

# Freeze the earlier layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(512, activation='relu'),  # Additional dense layer
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Train the model without the LearningRateScheduler
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]  # No lr_scheduler here
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f'Test Accuracy: {test_acc*100:.2f}%')

# Get predictions on the test set
y_pred = model.predict(test_generator, verbose=1)
y_pred = (y_pred > 0.5).astype(int)  # Set threshold at 0.5

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Save confusion matrix as an image
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_iv3.png')

# Show classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot training and validation graphs
plt.figure(figsize=(12, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.show()

# Calculate precision-recall curve to adjust the threshold if needed
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
optimal_threshold = thresholds[np.argmax(precision + recall)]
y_pred_optimal = (y_pred > optimal_threshold).astype(int)

# Recalculate confusion matrix with optimal threshold
cm_optimal = confusion_matrix(y_test, y_pred_optimal)

# Plot confusion matrix with optimal threshold
plt.figure(figsize=(6,6))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix (Optimal Threshold {optimal_threshold:.2f})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_optimal_threshold.png')
