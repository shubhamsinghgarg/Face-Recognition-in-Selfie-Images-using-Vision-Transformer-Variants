import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import timm

# Define transformations with data augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size to match model expected input
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size to match model expected input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load the dataset with augmented transformations
train_dataset = datasets.ImageFolder(root='./WSD_Dataset/train', transform=train_transform)
test_dataset = datasets.ImageFolder(root='./WSD_Dataset/test', transform=test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the pretrained model
model_name = 'vit_base_patch16_clip_224'
v = timm.create_model(model_name, pretrained=True, num_classes=42)

# Add dropout to the model (if not already present)
class CustomModel(nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(0.5)  # Add dropout layer with 50% probability

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x

v = CustomModel(v)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(v.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
v.to(device)

num_epochs = 200

for epoch in range(num_epochs):
    v.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = v(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    training_accuracy = 100 * correct / total
    training_loss = running_loss / len(train_loader)

    v.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = v(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    testing_accuracy = 100 * correct / total
    testing_loss = test_loss / len(test_loader)

    scheduler.step(testing_loss)  # Adjust the learning rate based on validation loss

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {training_loss:.4f}, Training Accuracy: {training_accuracy:.2f}%, Testing Loss: {testing_loss:.4f}, Testing Accuracy: {testing_accuracy:.2f}%')

# Save the model
torch.save(v.state_dict(), 'vit_model_wsd.pth')

# To load the model for future use
# v.load_state_dict(torch.load('vit_model_wsd.pth'))

