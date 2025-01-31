import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
import seaborn as sns

ImageFile.LOAD_TRUNCATED_IMAGES = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomRotation(30),  
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root='dataset_s/train', transform=transform)
test_data = datasets.ImageFolder(root='dataset_s/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"##############################{device}##############################")


class car_or_track(nn.Module):
    def __init__(self):
        super(car_or_track, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)     
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 14 * 14, 128)  
        self.bn_fc1 = nn.BatchNorm1d(128)  
        self.dropout = nn.Dropout(0.5) 

        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))  
        x = self.flatten(x)

        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = car_or_track().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00005 , weight_decay=0.01)
    class_weights = torch.tensor([1.1, 1.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    epochs = 100

    epoch_count = []
    loss_values = []
    test_loss_values = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_train_loss = running_loss / len(train_loader)
        loss_values.append(average_train_loss) 

        model.eval()
        with torch.inference_mode():
            total_test_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss = loss_fn(outputs, labels)
                total_test_loss += test_loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        average_test_loss = total_test_loss / len(test_loader)
        accuracy = 100 * correct / total

        epoch_count.append(epoch + 1)  
        test_loss_values.append(average_test_loss)  

        print(f"Epoch: {epoch+1} | Train Loss: {average_train_loss:.4f} | Test Loss: {average_test_loss:.4f} | Accuracy: {accuracy:.2f}%")

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print(f"Długość epoch_count: {len(epoch_count)}")
    print(f"Długość loss_values: {len(loss_values)}")
    print(f"Długość test_loss_values: {len(test_loss_values)}")

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_count, loss_values, label="Train Loss")
    plt.plot(epoch_count, test_loss_values, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()


    model.eval()
    test_samples = []
    pred_classes = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
        
            test_samples.extend(inputs.cpu())  
            pred_classes.extend(predicted.cpu())  
            test_labels.extend(labels.cpu())  

    class_names = train_data.classes
    plt.figure(figsize=(16, 16))
    nrows = 4
    ncols = 4
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
    
        sample = test_samples[i].numpy().transpose((1, 2, 0))  
        sample = np.clip(sample, 0, 1)  

        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]

        plt.imshow(sample)
        plt.axis('off')
    
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, color="green") 
        else:
            plt.title(title_text, fontsize=10, color="red") 

    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), 'model.pth')


    print(train_data.classes) 
    print(train_data.class_to_idx)  


    cm = confusion_matrix(test_labels, pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()