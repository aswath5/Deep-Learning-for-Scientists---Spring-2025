import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb  # Import wandb
import seaborn as sns

# Initialize WandB
wandb.init(project='malaria-classification', config={
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
    'optimizer': 'Adam'
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load dataset
train_df = pd.read_csv("/ocean/projects/cis240109p/shanmuga/dataset/train_data.csv")
train_df["path"] = train_df["img_name"].apply(lambda x: os.path.join("/ocean/projects/cis240109p/shanmuga/dataset/train_images", x))
print("Dataset Loaded")

# Train-test split
train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=42)
print("Train-validation split Done")

class Malaria(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["path"]
        label = self.dataframe.iloc[idx]["label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
print("Dataset class is defined")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomAffine(degrees=30, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

])
print("Data augmentation and preprocessing")
train_dataset = Malaria(train_data, transform=transform)
val_dataset = Malaria(val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("Loaders are created")

class MalariaCNN(nn.Module):
    def __init__(self):
        super(MalariaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)  # Binary classification

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.avgpool(x)  # Get feature embeddings here
        x = torch.flatten(features, 1)
        x = self.fc(x)
        return x, features
        

# Initialize model
model = MalariaCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Log the model architecture to WandB
wandb.watch(model, log="all")



def train_model(model, train_loader, val_loader, epochs=1):
    for epoch in range(epochs):
        model.train()
        train_loss, correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)  # Get logits (outputs) and ignore the feature embeddings
            loss = criterion(outputs, labels)  # Calculate loss using logits
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        # Calculate training metrics
        train_acc = correct / len(train_loader.dataset)
        train_loss_avg = train_loss / len(train_loader)
        
       

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)  # Get logits (outputs) and ignore the feature embeddings
                loss = criterion(outputs, labels)  # Calculate loss using logits
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        # Calculate validation metrics
        val_acc = val_correct / len(val_loader.dataset)
        val_loss_avg = val_loss / len(val_loader)
         # Log validation accuracy
        wandb.log({
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "epoch": epoch + 1
        })
        

        # Print training and validation metrics
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")



# Train model
train_model(model, train_loader, val_loader, epochs=10)

# Extract embeddings and visualize using t-SNE (optional)
# You can log images and predictions here using WandB for visualization

# Saving model weights to WandB (optional)
wandb.save("malaria_cnn_model.pth")


# Define test dataset class
class TestMalariaDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = sorted(os.listdir(image_folder))  # Ensure consistent order
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name  # No label since this is test data

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])



])

# Load test data
test_folder = "/ocean/projects/cis240109p/shanmuga/dataset/test_images"
test_dataset = TestMalariaDataset(test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
model.eval()  # Set model to evaluation mode

# Run inference and store predictions
predictions = []
with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        #triallll
        # predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()
        predicted_labels = torch.argmax(outputs[0], dim=1).cpu().numpy()

        
        for img_name, label in zip(img_names, predicted_labels):
            predictions.append((img_name, label))

# Create DataFrame and save CSV
submission_df = pd.DataFrame(predictions, columns=["img_name", "label"])
submission_df.to_csv("submission.csv", index=False)

 #Extract embeddings
embeddings = []
labels_emb = []

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        _, features = model(images)  # Get feature embeddings
        embeddings.append(features.view(features.size(0), -1).cpu())  # Flatten the features
        labels_emb.append(labels.cpu())

# Convert to numpy arrays
X = torch.cat(embeddings, dim=0).numpy()
y = torch.cat(labels_emb, dim=0).numpy()

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)  # This should now work with 2D data

# Create DataFrame for visualization
tsne_df = pd.DataFrame({
    "Component 1": X_2d[:, 0],
    "Component 2": X_2d[:, 1],
    "Label": y.astype(int)
})

# Plot using seaborn
plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df, x="Component 1", y="Component 2", hue="Label", palette=["blue", "orange"], s=10, alpha=0.7)
plt.title("t-SNE Visualization of Malaria CNN Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Label")
plt.savefig("t-SNE_malaria.png")
plt.show()
