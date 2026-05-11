import os
# Suppress TensorFlow logs to save memory/noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np
import gc
from PIL import Image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from huggingface_hub import hf_hub_download

# Lazy import tensorflow only when needed to save startup memory
def load_tf_model(path):
    import tensorflow as tf
    # Force CPU usage
    tf.config.set_visible_devices([], 'GPU')
    return tf.keras.models.load_model(path, compile=False)

# =========================
# Hugging Face Repo Details
# =========================
REPO_ID = "DONCHAN123/FlowerDetection"

_model = None
_index_to_class = None

def get_model_and_mapping():
    global _model, _index_to_class
    if _model is None:
        try:
            print("--- Loading Model (Memory Optimized Mode) ---")
            model_path = hf_hub_download(repo_id=REPO_ID, filename="V1.h5")
            _model = load_tf_model(model_path)

            pkl_path = hf_hub_download(repo_id=REPO_ID, filename="V1.pkl")
            with open(pkl_path, 'rb') as f:
                class_indices = pickle.load(f)
            _index_to_class = {v: k for k, v in class_indices.items()}

            print("--- Model Loaded ---")
            gc.collect()
        except Exception as e:
            print(f"!!! Model Load Error: {str(e)}")
            raise e
    return _model, _index_to_class

class FlowerDetectionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request):
        """Health Check for Browser Testing"""
        return Response({
            "status": "online",
            "info": "Render backend is alive. POST to this endpoint with an image."
        }, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            up_image = request.FILES['image']
            model, index_to_class = get_model_and_mapping()

            # Process image in a memory-efficient way
            with Image.open(up_image) as img:
                img = img.convert('RGB').resize((128, 128))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))

            # Clear temporary data immediately
            del img_array
            gc.collect()

            if int(predicted_index) in index_to_class:
                predicted_class = index_to_class[int(predicted_index)]
                return Response({
                    "flowerName": predicted_class,
                    "confidenceScore": confidence,
                    "description": f"This is a {predicted_class}. Identified successfully."
                }, status=status.HTTP_200_OK)

            return Response({"error": "Prediction failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        except Exception as e:
            print(f"Server Error: {str(e)}")
            return Response({"error": "Server is low on memory. Please try a smaller image."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
