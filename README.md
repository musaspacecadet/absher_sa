

## Directory Structure

```
model/
├── __init__.py: Initializes the model package, defining paths to model weights and labels.
├── dataset.zip: (Optional) A zipped archive of the image dataset. You will need to unzip this.
├── fetch_dataset.py: Downloads CAPTCHA images from a specified URL.
├── inference.py: Contains the model definition and inference pipeline for CAPTCHA prediction.
├── label_dataset.py: Provides a Gradio interface for labeling CAPTCHA images.
├── labels.json: Stores the labels for the CAPTCHA images in JSON format.
├── train.py: Trains the CAPTCHA solver model.
├── __pycache__/: Python cache directory.
└── weights/:
    └── model_0.99.pth: Pre-trained weights for the CAPTCHA solver model.
```

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision pillow requests tqdm gradio
    ```

2.  **(Optional) Extract Dataset:**
    If `dataset.zip` exists, unzip it into the `model/` directory:
    ```bash
    unzip model/dataset.zip -d model/
    ```

## Usage

### 1. Data Fetching

*   Run the `fetch_dataset.py` script to download CAPTCHA images:

    ```bash
    python model/fetch_dataset.py
    ```

    *   The number of images downloaded can be configured within the script.
    *   Images are saved to the `model/dataset` directory.

### 2. Data Labeling

*   Start the Gradio labeling interface by running `label_dataset.py`:

    ```bash
    python model/label_dataset.py
    ```

*   Open the provided URL in your browser.
*   Label the CAPTCHA images through the interface.
    *   **Important:** Labels must consist of 4 numeric characters.
*   Labels are automatically saved to `model/labels.json`.

### 3. Training

*   Ensure you have a labeled dataset in `model/labels.json`.
*   Train the model using the `train.py` script:

    ```bash
    python model/train.py
    ```

*   Training progress and validation accuracy are printed to the console.
*   Plots of training/validation loss and validation accuracy are generated and saved as `train.png`.
*   The best model weights are saved to the `model/weights` directory, with the validation accuracy included in the filename.

### 4. Inference

*   The `inference.py` script demonstrates how to load the pre-trained model and perform inference on a single image.

    *   Modify the `image_path_to_predict` variable within the `if __name__ == "__main__":` block to point to the image you want to predict.

    ```bash
    python model/inference.py
    ```

*   The predicted CAPTCHA text will be printed to the console.
*   The `create_inference_pipeline` function in `model/inference.py` can be used to create a reusable inference function for integration into other applications.

## Files

*   `model/__init__.py`: Defines paths to model weights and labels file for easy access.
*   `model/fetch_dataset.py`: Downloads CAPTCHA images from a website. Requires internet access.
*   `model/inference.py`: Contains the `CaptchaSolver` model definition and the `create_inference_pipeline` function for creating a prediction pipeline. Includes a CTC decoding function.
*   `model/label_dataset.py`: Provides a Gradio interface for labeling images. Uses the trained model to provide prediction hints.
*   `model/labels.json`: Stores the image paths and their corresponding labels in JSON format.
*   `model/train.py`: Trains the `CaptchaSolver` model using the labeled dataset. Includes data loading, preprocessing, model definition, training loop, validation, and saving the best model weights. Also includes a modified CTC decoding function.
*   `model/weights/model_0.99.pth`: Pre-trained model weights.

## Notes

*   The `fetch_dataset.py` script uses a specific URL and headers to download CAPTCHA images. This may need to be updated if the source website changes.
*   The `labels.json` file should contain the path to each image relative to the script's location.
*   The `train.py` script saves the model weights with the validation accuracy in the filename.
*   The `ctc_decode` function in `inference.py` and `train.py` is crucial for handling repeating characters in the CAPTCHA labels.
*   The model architecture and training parameters can be adjusted in the `train.py` script to improve performance.
