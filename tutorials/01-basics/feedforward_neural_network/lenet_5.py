
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net= nn.Sequential(
            # 1 input image channel, 6 output channels, 5x5 square convolution
            # kernel
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, 5), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(), 
            nn.Linear(16 * 5 * 5, 120),  # 5*5 from image dimension 
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        logits= self.net(x)
        return logits

model= LeNet5()
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in model.net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)    

model= LeNet5().to(device)
# Hyper-parameters 
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='C:/Library/Applications/Python/data/MachineLearning/ml/Stats/Materials/data/pytorch', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='C:/Library/Applications/Python/data/MachineLearning/ml/Stats/Materials/data/pytorch', 
                                          train=False, 
                                          transform=transforms.ToTensor())


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)  

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)
        total += labels.size(0)
        correct += (y_pred == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt') #'../../data'

