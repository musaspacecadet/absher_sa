import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

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

def export_to_onnx(model_path, img_width, img_height, onnx_path):
    """
    Exports the PyTorch model to ONNX format.

    Args:
        model_path (str): Path to the saved PyTorch model.
        img_width (int): Width of the input image.
        img_height (int): Height of the input image.
        onnx_path (str): Path to save the ONNX model.
    """

    # Initialize the model
    model = CaptchaSolver(img_height, img_width)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()  # Set the model to evaluation mode

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, img_height, img_width)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'sequence_length', 1: 'batch_size'}}
    )
    print(f"Model exported to {onnx_path}")


if __name__ == "__main__":
    # Define paths and parameters
    model_path = "weights/model_0.99.pth"  # Path to your PyTorch model weights
    img_width = 200
    img_height = 50
    onnx_path = "weights/captcha_solver.onnx"  # Output ONNX file path

    # Export the model
    export_to_onnx(model_path, img_width, img_height, onnx_path)