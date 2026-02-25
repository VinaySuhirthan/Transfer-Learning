# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

This experiment demonstrates transfer learning using a pre-trained ResNet18 model on a custom image dataset. Instead of training a deep neural network from scratch, the pre-trained model’s feature extraction layers are reused, and only the final classification layer is retrained. This approach reduces training time, requires less data, and achieves high accuracy.

## DESIGN STEPS
### STEP 1:
Data Preprocessing – Resize all images to 224×224 and convert them into tensors suitable for ResNet input.

### STEP 2: 
Dataset Loading – Organize images into train/test sets and load them using ImageFolder and DataLoader.

### STEP 3:
Load Pretrained Model – Use ResNet18 trained on ImageNet as the base model.

### STEP 4:
Modify Final Layer – Freeze earlier layers and replace the fully connected layer to match the number of dataset classes.

### STEP 5:
Train and Evaluate – Train only the final layer, then test the model and analyze results using a confusion matrix and classification report.

## PROGRAM
```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer to match the dataset classes
for param in model.parameters():
    param.requires_grad = False   # freeze earlier layers
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

```
```python
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: K S Vinay Suhirthan")
    print("Register Number:  212224230305")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="893" height="702" alt="image" src="https://github.com/user-attachments/assets/d2032768-d7d3-440b-b961-f857f4c0eb59" />
<img width="253" height="58" alt="image" src="https://github.com/user-attachments/assets/dc274465-2e80-4d49-9269-60d0f2987e3e" />



### Confusion Matrix
<img width="716" height="465" alt="image" src="https://github.com/user-attachments/assets/8fb3b38e-970e-4b14-99e4-7615728a9517" />


### Classification Report

<img width="464" height="207" alt="image" src="https://github.com/user-attachments/assets/48d1ac40-75a1-4ac2-8a26-7b337421e7c9" />


### New Sample Prediction
<img width="524" height="397" alt="image" src="https://github.com/user-attachments/assets/cfac4b19-af85-4318-bd9f-ae7f75e79e9f" />

<img width="573" height="404" alt="image" src="https://github.com/user-attachments/assets/5101e10a-259c-40dc-88e8-85ad4f1fcaef" />


## RESULT
The Implementation of Transfer Learning for classification using VGG-19 architecture is successful.
