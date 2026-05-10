import os
import pickle
import numpy as np
from PIL import Image
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging

logger = logging.getLogger(__name__)

# =========================
# Hugging Face Repo Details
# =========================
REPO_ID = "DONCHAN123/FlowerDetection"

# Global variables to cache model and class mapping
_model = None
_index_to_class = None

def get_model_and_mapping():
    global _model, _index_to_class
    if _model is None:
        try:
            print("--- Starting Model Download from Hugging Face ---")
            logger.info("Downloading and loading model...")

            model_path = hf_hub_download(repo_id=REPO_ID, filename="V1.h5")
            print(f"Model downloaded to: {model_path}")
            _model = load_model(model_path)

            pkl_path = hf_hub_download(repo_id=REPO_ID, filename="V1.pkl")
            print(f"Mapping downloaded to: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                class_indices = pickle.load(f)
            _index_to_class = {v: k for k, v in class_indices.items()}
            print("--- Model and Mapping Loaded Successfully ---")
        except Exception as e:
            print(f"!!! Error Loading Model: {str(e)}")
            raise e
    return _model, _index_to_class

class FlowerDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request):
        return Response({"status": "Server is running. Send a POST request with an image."}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 1. Get the image from the request
            up_image = request.FILES['image']
            print(f"Received image: {up_image.name}")

            # 2. Load Model and Mapping
            # This part takes the most time on Render (Cold Start)
            model, index_to_class = get_model_and_mapping()

            # 3. Preprocess Image
            img = Image.open(up_image)
            img = img.convert('RGB')
            img = img.resize((128, 128))

            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 4. Predict
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))

            # 5. Get Result
            if int(predicted_index) in index_to_class:
                predicted_class = index_to_class[int(predicted_index)]
                description = f"This is a {predicted_class}. (Description can be expanded with more data)."

                print(f"Prediction: {predicted_class} ({confidence:.2f})")

                return Response({
                    "flowerName": predicted_class,
                    "confidenceScore": confidence,
                    "description": description
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    "error": f"Predicted index {predicted_index} not found in class mapping."
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            print(f"Prediction Error: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
