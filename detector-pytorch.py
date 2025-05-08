from google.colab import files
from google.colab import drive
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d emmarex/plantdisease
!unzip -o plantdisease.zip


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
dataset = datasets.ImageFolder('/content/plantvillage/PlantVillage', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader=DataLoader(test_data,batch_size=64, shuffle=False )
train_data[0][0].shape

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1=nn.Conv2d(3,64, kernel_size=3)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(64,128, kernel_size=3)
        self.pool2=nn.MaxPool2d(2,2)
        self.dropout=nn.Dropout(0.3)
        self.fc1=nn.Linear(128*14*14,512)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(512,128)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(128,64)
        self.relu3=nn.ReLU()
        self.fc4=nn.Linear(64,38)
    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=x.view(x.size(0), -1)
        x=self.dropout(self.relu1(self.fc1(x)))
        x=self.dropout(self.relu2(self.fc2(x)))
        x=self.dropout(self.relu3(self.fc3(x)))
        x=self.fc4(x)
        return x

model=Model().to(device)

optimizer=optim.Adam(model.parameters(), lr=0.001)
criterion=nn.CrossEntropyLoss()
epochs=20

for epoch in range(epochs):
  current_loss=0
  for i, (inputs, labels) in enumerate(train_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs=model(inputs)
    loss=criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    current_loss += loss.item()

  print(f"Epoch: {epoch}, Loss: {current_loss / len(train_loader)}")


drive.mount('/content/drive',force_remount=True)
torch.save(model.state_dict(), '/content/drive/MyDrive/plant_disease_model.pth')


from google.colab import drive
drive.mount('/content/drive',force_remount=True)
model.load_state_dict(torch.load('/content/drive/MyDrive/plant_disease_model.pth'))

model.eval()
with torch.no_grad():
  correct=0
  total=0
  for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs=model(inputs)
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)
accuracy = 100 * correct / total
accuracy=round(accuracy)
print(accuracy,'%')

files.upload()


class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]





transform2 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

with torch.no_grad():
    output = model(pred_image)
    _, predicted = torch.max(output, 1)
    pred_label = predicted.item()
    print(f'Predicted Label: {class_names[pred_label]}')
