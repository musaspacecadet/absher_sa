# Define paths and parameters
#from model.inference import create_inference_pipeline
from model.test_onxx_inference import create_onnx_inference_pipeline as create_inference_pipeline
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths
model_path = os.path.join(script_dir, "weights/captcha_solver.onnx")
labels_json_path = os.path.join(script_dir, "labels.json")


