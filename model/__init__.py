# Define paths and parameters
from model.inference import create_inference_pipeline
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths
model_path = os.path.join(script_dir, "weights/model_0.99.pth")
labels_json_path = os.path.join(script_dir, "labels.json")


