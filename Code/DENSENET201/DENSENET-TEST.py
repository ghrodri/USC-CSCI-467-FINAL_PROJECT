import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# Define the custom attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)  # Accept any extra arguments
    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            name="attention_weights", shape=(input_shape[-1], 1), initializer="random_normal"
        )
    def call(self, inputs):
        attention_scores = tf.nn.softmax(tf.matmul(inputs, self.attention_weights), axis=1)
        weighted_input = inputs * attention_scores
        return weighted_input

    def get_config(self):
        # Save the custom configuration
        base_config = super(AttentionLayer, self).get_config()
        return base_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
register_keras_serializable(package='Custom', name='AttentionLayer')
# Load the model with the custom layer
model = load_model('best_model.keras', custom_objects={"AttentionLayer": AttentionLayer})
# Path of your test data
test_dir = "/Users/ghrodri17/Downloads/FINAL_DATA/Test_Images"
# Create the test image generator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False 
)
# Make predictions on the test set
predictions = model.predict(test_generator, verbose=1)
# Get true labels and predicted class labels
true_labels = test_generator.classes
predicted_labels = np.argmax(predictions, axis=1)
# Print the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)  
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Print the classification report
report = classification_report(true_labels, predicted_labels, target_names=test_generator.class_indices.keys())
print("\nClassification Report:\n")
print(report)
