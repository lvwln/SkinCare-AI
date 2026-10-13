import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. Define paths and settings
data_dir = "data/train"
batch_size = 16
num_epochs = 5  # Set to 5 epochs for a quick training run on a laptop

# 2. Setup image transformations (resize and normalize for ResNet)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load dataset automatically from your folders
print("Loading images from folders...")
image_dataset = datasets.ImageFolder(data_dir, data_transforms)
dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

class_names = image_dataset.classes
print(f"Found classes: {class_names}")

# 4. Load a pre-trained ResNet18 model and adjust the final layer for our number of classes
print("Initializing model...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
print("Starting model training...")
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(image_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

print("Training complete!")

# 6. Save the trained model weights into the models folder
os.makedirs("models", exist_ok=True)
model_path = "models/skin_classifier.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, model_path)

print(f"Model successfully saved to {model_path}!")