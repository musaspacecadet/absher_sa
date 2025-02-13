import onnxruntime
import numpy as np
from PIL import Image

def preprocess_image(image_path, img_width, img_height):
    """Preprocesses the image for ONNX Runtime."""
    image = Image.open(image_path).convert("RGB")
    # Resize
    image = image.resize((img_width, img_height))
    # Convert to numpy array
    image = np.array(image)
    # Normalize (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image / 255.0 - mean) / std
    # Add batch dimension and change to (batch, channel, height, width)
    image = image.transpose(2, 0, 1).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def ctc_decode(pred_sequence):
    """CTC decoding (same as before)."""
    previous = pred_sequence[0]
    decoded = [previous] if previous != 0 else []
    for current in pred_sequence[1:]:
        if current != 0:
            if current != previous:
                decoded.append(current)
            elif len(decoded) > 0 and decoded[-1] != current:
                decoded.append(current)
        previous = current
    return decoded

def postprocess_output(output):
    """Postprocesses the ONNX Runtime output."""
    # Softmax (along the character dimension)
    exp_output = np.exp(output - np.max(output, axis=2, keepdims=True))  # Numerical stability
    softmax_output = exp_output / np.sum(exp_output, axis=2, keepdims=True)

    # Get predicted indices
    predicted = np.argmax(softmax_output, axis=2)
    predicted = predicted.transpose(1, 0)  # Transpose to (batch, seq_len)

    # CTC decode
    raw_prediction = predicted[0]  # First (and only) batch
    decoded_prediction = ctc_decode(raw_prediction)

    # Map to characters
    idx_to_char = {0: '#', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4',
                   6: '5', 7: '6', 8: '7', 9: '8', 10: '9'}
    predicted_label = [idx_to_char[idx] for idx in decoded_prediction]
    return "".join(predicted_label)


def create_onnx_inference_pipeline(onnx_path, img_width, img_height):
    """
    Creates an inference pipeline using ONNX Runtime.

    Args:
        onnx_path (str): Path to the ONNX model.
        img_width (int): Width of the input image.
        img_height (int): Height of the input image.

    Returns:
        A function that takes an image path and returns the predicted CAPTCHA text.
    """

    # Load the ONNX model (do this once, outside the prediction function)
    ort_session = onnxruntime.InferenceSession(onnx_path)

    def predict_captcha(image_path):
        """
        Predicts the CAPTCHA text from an image using ONNX Runtime.

        Args:
            image_path (str): Path to the CAPTCHA image.

        Returns:
            str: The predicted CAPTCHA text.
        """

        # Preprocess the image
        input_image = preprocess_image(image_path, img_width, img_height)

        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_image}
        ort_outs = ort_session.run(None, ort_inputs)

        # Postprocess the output
        predicted_text = postprocess_output(ort_outs[0])
        return predicted_text

    return predict_captcha

if __name__ == "__main__":
    onnx_model_path = "weights/captcha_solver.onnx"  # Path to your ONNX model
    image_path = "../captcha.jpeg"  # Path to the CAPTCHA image
    img_width = 200
    img_height = 50

    # Create the inference pipeline
    predict_fn = create_onnx_inference_pipeline(onnx_model_path, img_width, img_height)

    # Predict the CAPTCHA text
    predicted_text = predict_fn(image_path)
    print(f"Predicted CAPTCHA: {predicted_text}")

    # Example with a different image (no need to reload the model)
    another_image_path = "../another_captcha.jpeg"  # Replace with another image
    if another_image_path != image_path: # avoid FileNotFoundError
        try:
            another_predicted_text = predict_fn(another_image_path)
            print(f"Predicted CAPTCHA (another image): {another_predicted_text}")
        except FileNotFoundError:
            print(f"Image file not found: {another_image_path}")