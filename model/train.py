import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Path to the JSON file with image paths and labels
labels_json_path = "labels.json"
# Image dimensions
img_width = 200
img_height = 50

# Load the labels from the JSON file
with open(labels_json_path, "r") as f:
    data = json.load(f)

# Extract the image paths and labels from the JSON
images = list(data.keys())
labels = list(data.values())

# --- Data Preprocessing ---

# Define the character set (all unique characters in your labels)
characters = set()
for label in labels:
    characters.update(list(label))
characters = sorted(list(characters))
n_classes = len(characters)

# Create a mapping from character to index and vice versa
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}  # Shift indices by 1
char_to_idx['#'] = 0  # Use '#' as the blank symbol
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

class CaptchaDataset(Dataset):
    def __init__(self, image_paths, labels, width, height, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.width = width
        self.height = height
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        label_indices = [char_to_idx[char] for char in label]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return img, label_tensor

# --- Data Augmentation and Transformations ---
train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Data Splitting ---
train_size = int(0.8 * len(images))
val_size = len(images) - train_size
train_images, val_images = torch.utils.data.random_split(images, [train_size, val_size])
train_labels = [labels[i] for i in train_images.indices]
val_labels = [labels[i] for i in val_images.indices]
train_images = [images[i] for i in train_images.indices]
val_images = [images[i] for i in val_images.indices]

train_dataset = CaptchaDataset(train_images, train_labels, img_width, img_height, transform=train_transform)
val_dataset = CaptchaDataset(val_images, val_labels, img_width, img_height, transform=val_transform)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class CaptchaSolver(nn.Module):
    def __init__(self, num_chars, img_height, img_width):
        super(CaptchaSolver, self).__init__()
        self.num_chars = num_chars
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

# --- Modified CTC Decoding Function ---
def ctc_decode(pred_sequence):
    """
    Modified CTC decoding that properly handles repeating characters
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CaptchaSolver(n_classes + 1, img_height, img_width).to(device)

criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        label_lengths = torch.full((labels.size(0),), labels.size(1), dtype=torch.long)
        input_lengths = torch.full((images.size(0),), model.feature_map_width, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)
        log_probs = F.log_softmax(outputs, dim=2)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            label_lengths = torch.full((labels.size(0),), labels.size(1), dtype=torch.long)
            input_lengths = torch.full((images.size(0),), model.feature_map_width, dtype=torch.long)

            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=2)

            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            running_loss += loss.item()

            _, predicted = log_probs.max(2)
            predicted = predicted.transpose(0, 1)
            
            for j in range(predicted.size(0)):
                raw_prediction = predicted[j].cpu().numpy()
                decoded_prediction = ctc_decode(raw_prediction)  # Use the modified decoder
                
                true_label = [idx_to_char[idx.item()] for idx in labels[j] if idx.item() != 0]
                predicted_label = [idx_to_char[idx] for idx in decoded_prediction]

                if "".join(predicted_label) == "".join(true_label):
                    correct_predictions += 1
                total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return running_loss / len(val_loader), accuracy

# --- Training Loop ---
num_epochs = 200
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# --- Plotting Training and Validation Metrics ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy')

plt.tight_layout()
plt.show()
plt.savefig("train.png")

# --- Inference Function ---
def predict_captcha(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        log_probs = F.log_softmax(output, dim=2)
        _, predicted = log_probs.max(2)
        predicted = predicted.transpose(0, 1)

        raw_prediction = predicted[0].cpu().numpy()
        decoded_prediction = ctc_decode(raw_prediction)  # Use the modified decoder
        predicted_label = [idx_to_char[idx] for idx in decoded_prediction]
        
        return "".join(predicted_label)

def evaluate_and_show_predictions(model, val_loader, transform, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    results = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=2)
            _, predicted = log_probs.max(2)
            predicted = predicted.transpose(0, 1)

            for j in range(predicted.size(0)):
                raw_prediction = predicted[j].cpu().numpy()
                decoded_prediction = ctc_decode(raw_prediction)  # Use the modified decoder
                
                true_label = [idx_to_char[idx.item()] for idx in labels[j] if idx.item() != 0]
                predicted_label = [idx_to_char[idx] for idx in decoded_prediction]

                predicted_text = "".join(predicted_label)
                actual_text = "".join(true_label)

                if predicted_text == actual_text:
                    correct_predictions += 1
                total_predictions += 1

                results.append((images[j].cpu(), actual_text, predicted_text))

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Display example predictions
    num_examples_to_show = 15
    for img, actual, predicted in random.sample(results, num_examples_to_show):
        img = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.imshow(img)
        plt.title(f"Actual: {actual}, Predicted: {predicted}")
        plt.show()

# Save the model
torch.save(model.state_dict(), f"weights/model_{val_accuracy:.2f}.pth")

# Evaluate on validation set
evaluate_and_show_predictions(model, val_loader, val_transform, device)