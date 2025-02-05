import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class CaptchaSolver(nn.Module):
    def __init__(self, img_height, img_width):
        super(CaptchaSolver, self).__init__()
        self.num_chars = 11  # 10 digits + 1 blank (#)
        self.img_height = img_height
        self.img_width = img_width

        # CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.feature_map_height = self.img_height // (2 * 2 * 2)
        self.feature_map_width = self.img_width // (2 * 2 * 2)

        # RNN layers
        self.rnn_input_size = 128 * self.feature_map_height
        self.rnn_hidden_size = 256
        self.rnn = nn.GRU(self.rnn_input_size, self.rnn_hidden_size, bidirectional=True, batch_first=True)

        # Output layer
        self.fc = nn.Linear(self.rnn_hidden_size * 2, self.num_chars)

    def forward(self, x):
        # CNN
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Reshape for RNN
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)

        # RNN
        x, _ = self.rnn(x)

        # Output layer
        x = self.fc(x)
        x = x.permute(1, 0, 2)

        return x

def ctc_decode(pred_sequence):
    """
    CTC decoding that properly handles repeating characters.
    
    Args:
        pred_sequence: numpy array of predicted indices
        
    Returns:
        list: Decoded sequence of indices with proper handling of duplicates
    """
    previous = pred_sequence[0]
    decoded = [previous] if previous != 0 else []  # Skip blank token (0)
    
    for current in pred_sequence[1:]:
        if current != 0:  # Skip blank token
            if current != previous:  # Always add if different from previous
                decoded.append(current)
            elif len(decoded) > 0 and decoded[-1] != current:  # Add repeating char if not immediately consecutive
                decoded.append(current)
        previous = current
    
    return decoded

def create_inference_pipeline(model_path, img_width, img_height):
    """
    Creates an elegant inference pipeline for the CAPTCHA solver with proper CTC decoding.

    Args:
        model_path (str): Path to the saved model state dictionary.
        img_width (int): Width of the input images.
        img_height (int): Height of the input images.

    Returns:
        A function that takes an image path and returns the predicted CAPTCHA text.
    """

    

    # Blank symbol -> #
    # 1. Numerical character mapping
    idx_to_char = {0: '#', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 
                   6: '5', 7: '6', 8: '7', 9: '8', 10: '9'}
    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CaptchaSolver(img_height, img_width).to(device)

    # 3. Load Model Weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 4. Define Image Transformation
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 5. Inference Function
    def predict_captcha(image_path):
        """
        Predicts the CAPTCHA text from an image using proper CTC decoding.

        Args:
            image_path (str): Path to the CAPTCHA image.

        Returns:
            str: The predicted CAPTCHA text.
        """
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            log_probs = F.log_softmax(output, dim=2)
            _, predicted = log_probs.max(2)
            predicted = predicted.transpose(0, 1)

            raw_prediction = predicted[0].cpu().numpy()
            decoded_prediction = ctc_decode(raw_prediction)  # Use the new CTC decoder
            predicted_label = [idx_to_char[idx] for idx in decoded_prediction]
            
            return "".join(predicted_label)

    return predict_captcha

# --- Example Usage ---
if __name__ == "__main__":
    # Define paths and parameters
    model_path = "weights/model_0.99.pth"
    img_width = 200
    img_height = 50

    # Create the inference pipeline
    predict_fn = create_inference_pipeline(model_path, img_width, img_height)

    # Example usage with a single image
    image_path_to_predict = "../captcha.jpeg"
    predicted_text = predict_fn(image_path_to_predict)
    print(f"Predicted CAPTCHA: {predicted_text}")