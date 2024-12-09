# Clothing Recommendation Module
# This module uses a pre-trained InceptionResNetV2 model to classify clothing and extract attributes for recommendations.

# Step 1: Import Required Libraries
import os
import cv2
import numpy as np
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

# Step 2: Load Pre-Trained Model
base_model = InceptionResNetV2(weights="imagenet", include_top=True)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)

# Step 3: Preprocessing Function
def preprocess_image(image_path):
    """Preprocesses an image for the model input."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (299, 299))  # InceptionResNetV2 requires 299x299 input size
    image = preprocess_input(image)  # Normalize for InceptionResNetV2
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Step 4: Predict Clothing Type
def predict_clothing_type(image_path):
    """Predicts clothing type using the pre-trained model."""
    image = preprocess_image(image_path)
    predictions = base_model.predict(image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return [(pred[1], pred[2]) for pred in decoded_predictions]

# Step 5: Extract Features for Recommendations
def extract_clothing_features(image_path):
    """Extracts features from the image for recommendation purposes."""
    image = preprocess_image(image_path)
    features = feature_extractor.predict(image)
    return features

# Step 6: Recommendation System Based on Predicted Clothing Type
def recommend_outfits(predictions):
    """Provide a recommendation based on the predicted clothing type."""
    clothing_type = predictions[0][0].lower()
    
    recommendations = {
        't-shirt': "We recommend pairing this T-shirt with casual jeans or shorts for a comfortable look.",
        'dress': "Pair this dress with a stylish cardigan or jacket and some high heels for a chic outfit.",
        'sweater': "A sweater goes perfectly with skinny jeans or a skirt and boots for a cozy outfit.",
        'jean': "These jeans would look great with a simple t-shirt or a button-up shirt.",
        'jacket': "You can layer this jacket over a t-shirt or dress for a trendy and warm look.",
        'skirt': "Pair this skirt with a tucked-in blouse or a casual top for a balanced look."
    }
    
    # Default recommendation if clothing type is not in the dictionary
    default_recommendation = "We recommend pairing this item with something stylish based on your taste."
    
    # Get the recommendation based on the clothing type
    return recommendations.get(clothing_type, default_recommendation)

# Step 7: Main Function Updated
def main():
    image_path = "C:\\Users\\hp\\OneDrive\\Desktop\\FashionMate\\FashionMate\\jacket.jpg"  # Replace with the actual image path
    
    # Predict a single image
    print("Single Image Prediction:")
    predictions = predict_clothing_type(image_path)
    for clothing_type, confidence in predictions:
        print(f"{clothing_type}: {confidence*100:.2f}%")
    
    # Extract features and recommend outfits based on prediction
    recommendation = recommend_outfits(predictions)
    print("\nRecommendation:")
    print(recommendation)

# Run the updated module
if __name__ == "__main__":
    main()
