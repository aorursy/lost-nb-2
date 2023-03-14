#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

DATA_DIR = Path("/kaggle/input")
if (DATA_DIR / "ucfai-core-fa19-cnns").exists():
    DATA_DIR /= "ucfai-core-fa19-cnns"
elif DATA_DIR.exists():
    # no-op to keep the proper data path for Kaggle
    pass
else:
    # You'll need to download the data from Kaggle and place it in the `data/`
    #   directory beside this notebook.
    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-cnns/data
    DATA_DIR = Path("data")


# In[2]:


# standard imports (Numpy, Pandas, Matplotlib)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# PyTorch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision import transforms
from torchsummary import summary

# Extras
import time
import os
import glob


# In[3]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
random.seed(42)


# In[4]:


input_size = (224,224)
batch_size = 32
num_workers = 4


# In[5]:


data_transforms = {
    'Train': transforms.Compose([transforms.Resize(input_size),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ]),
    'Validation': transforms.Compose([transforms.Resize(input_size),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ]),
    'Test': transforms.Compose([transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ])
}


# In[6]:


image_datasets = {
    x: ImageFolder(os.path.join(DATA_DIR, x),data_transforms[x])
    for x in ['Train', 'Validation']
}

# dataset class to load images with no labels, for our testing set to submit to
#   the competition
class ImageLoader(Dataset):
    def __init__(self, root, transform=None):
        # get image file paths
        self.images = sorted(
            glob.glob(os.path.join(root, "*")),
            key=self.glob_format
        )
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)
            return img
        else:
            return transforms.ToTensor(img)
        
    @staticmethod
    def glob_format(key):     
        key = key.split("/")[-1].split(".")[0]     
        return "{:04d}".format(int(key))
    
image_datasets['Test'] = ImageLoader(
    str(DATA_DIR / "Test"),
    transform=data_transforms["Test"]
)


# In[7]:


dataloaders = {
    x: DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    for x in ['Train', 'Validation']
}

test_loader = DataLoader(
    dataset=image_datasets['Test'],
    batch_size=1,
    shuffle=False
)


# In[8]:


dog_breeds = image_datasets['Train'].classes
print(dog_breeds)


# In[9]:


# Just printing the number of images in each dataset we created

dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation', 'Test']}

print('Train Length: {} | Valid Length: {} | Test Length: {}'.format(
    dataset_sizes['Train'], 
    dataset_sizes['Validation'],
    dataset_sizes['Test']
))


# In[10]:


# Here we're defining what component we'll use to train this model
# We want to use the GPU if available, if not we use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[11]:


# Plots a given number of images from a PyTorch Data
def show_random_imgs(num_imgs):
    for i in range(num_imgs):
        # We're plotting images from the training set
        train_dataset = image_datasets['Train']
        
        # Choose a random image
        rand = np.random.randint(0, len(train_dataset) + 1)
        
        # Read in the image
        ex = img.imread(train_dataset.imgs[rand][0])
        
        # Get the image's label
        breed = dog_breeds[train_dataset.imgs[rand][1]]
        
        # Show the image and print out the image's size
        #   (really the shape of it's array of pixels)
        plt.imshow(ex)
        print('Image Shape: ' + str(ex.shape))
        plt.axis('off')
        plt.title(breed)
        plt.show()
        

# Plots a batch of images served up by PyTorch    
def show_batch(batch):
    # Undo the transformations applied to the images when loading a batch
    batch = batch.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    batch = std * batch + mean
    batch = np.clip(batch, 0, 1)
    
    # Plot the batch
    plt.axis('off')
    plt.imshow(batch)
    
    # pause a bit so that plots are updated
    plt.pause(0.001)


# In[12]:


show_random_imgs(3)


# In[13]:


# Get a batch of training data (32 random images)
imgs, classes = next(iter(dataloaders['Train']))

# This PyTorch function makes a grid of images from a batch for us
batch = torchvision.utils.make_grid(imgs)

show_batch(batch)


# In[14]:


# It is good practice to maintain input dimensions as the image is passed
#   through convolution layers
# With a default stride of 1, and no padding, a convolution will reduce image
#   dimenions to:
#     out = in - m + 1, where _m_ is the size of the kernel and _in_ is a
#        dimension of the input

# Use this function to calculate the padding size neccessary to create an output
#   of desired dimensions

def get_padding(input_dim, output_dim, kernel_size, stride):
    # Calculates padding necessary to create a certain output size,
    # given a input size, kernel size and stride
    padding = (((output_dim - 1) * stride) - input_dim + kernel_size) // 2
  
    if padding < 0:
        return 0
    else:
        return padding


# In[15]:


# Make sure you calculate the padding amount needed to maintain the spatial
#   size of the input after each Conv layer

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # nn.Sequential() is simply a container that groups layers into one object
        # Pass layers into it separated by commas
        self.block1 = nn.Sequential(
            
            # The first convolutional layer. Think about how many channels the
            #   input starts off with
            # Let's have this first layer extract 32 features
            # YOUR CODE HERE
            raise NotImplementedError()
            
            # Don't forget to apply a non-linearity
            # YOUR CODE HERE
            raise NotImplementedError()
        
        self.block2 =  nn.Sequential(
            
            # The second convolutional layer. How many channels does it receive,
            #   given the number of features extracted by the first layer?
            # Have this layer extract 64 features
            # YOUR CODE HERE
            raise NotImplementedError()
            
            # Non linearity
            # YOUR CODE HERE
            raise NotImplementedError()
            
            # Lets introduce a Batch Normalization layer
            # YOUR CODE HERE
            raise NotImplementedError()
            
            # Downsample the input with Max Pooling
            # YOUR CODE HERE
            raise NotImplementedError()
        )
        
        # Mimic the second block here, except have this block extract 128
        #   features
        self.block3 =  nn.Sequential(
            # YOUR CODE HERE
            raise NotImplementedError()
        )
        
        # Applying a global pooling layer
        # Turns the 128 channel rank 4 tensor into a rank 2 tensor of size
        #   32 x 128 (32 128-length arrays, one for each of the inputs in a
        #   batch)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128, 512)
        
        # Introduce dropout to reduce overfitting
        self.drop_out = nn.Dropout(0.5)
        
        # Final fully connected layer creates the prediction array
        self.fc2 = nn.Linear(512, len(dog_breeds))
    
    # Feed the input through each of the layers we defined 
    def forward(self, x):
        
        # Input size changes from (32 x 3 x 224 x 224) to (32 x 32 x 224 x 224)
        x = self.block1(x)
        
        # Size changes from (32 x 32 x 224 x 224) to (32 x 64 x 112 x 112)
        #   after max pooling
        x = self.block2(x)
        
        # Size changes from (32 x 64 x 112 x 112) to (32 x 128 x 56 x 56)
        #   after max pooling
        x = self.block3(x)
        
        # Reshapes the input from (32 x 128 x 56 x 56) to (32 x 128)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer, size changes from (32 x 128) to (32 x 512)
        x = self.fc1(x)
        x = self.drop_out(x)
        
        # Size change from (32 x 512) to (32 x 133) to create prediction arrays
        #   for each of the images in the batch
        x = self.fc2(x)
        
        return x


# In[16]:


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 5

model.to(device)
summary(model, (3, 224, 224))


# In[17]:


def run_epoch(epoch, model, optimizer, dataloaders, device, phase):
  
    running_loss = 0.0
    running_corrects = 0

    if phase == 'Train':
        model.train()
    else:
        model.eval()

    # Looping through batches
    for i, (inputs, labels) in enumerate(dataloaders[phase]):
    
        # ensures we're doing this calculation on our GPU if possible
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero parameter gradients
        optimizer.zero_grad()
    
        # Calculate gradients only if we're in the training phase
        with torch.set_grad_enabled(phase == 'Train'):
      
            # This calls the forward() function on a batch of inputs
            outputs = model(inputs)

            # Calculate the loss of the batch
            loss = criterion(outputs, labels)

            # Gets the predictions of the inputs (highest value in the array)
            _, preds = torch.max(outputs, 1)

            # Adjust weights through backpropagation if we're in training phase
            if phase == 'Train':
                loss.backward()
                optimizer.step()

        # Document statistics for the batch
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    # Calculate epoch statistics
    epoch_loss = running_loss / image_datasets[phase].__len__()
    epoch_acc = running_corrects.double() / image_datasets[phase].__len__()

    return epoch_loss, epoch_acc


# In[18]:


def train(model, criterion, optimizer, num_epochs, dataloaders, device):
    start = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t| Epoch Time |')
    print('-' * 86)
    
    # Iterate through epochs
    for epoch in range(num_epochs):
        
        epoch_start = time.time()
       
        # Training phase
        train_loss, train_acc = run_epoch(epoch, model, optimizer, dataloaders, device, 'Train')
        
        # Validation phase
        val_loss, val_acc = run_epoch(epoch, model, optimizer, dataloaders, device, 'Validation')
        
        epoch_time = time.time() - epoch_start
           
        # Print statistics after the validation phase
        print("| {}\t | {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.0f}m {:.0f}s     |"
                      .format(epoch + 1, train_loss, train_acc, val_loss, val_acc, epoch_time // 60, epoch_time % 60))

        # Copy and save the model's weights if it has the best accuracy thus far
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    total_time = time.time() - start
    
    print('-' * 74)
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    print('Best validation accuracy: {:.4f}'.format(best_acc))

    # load best model weights and return them
    model.load_state_dict(best_model_wts)
    return model


# In[19]:


def test_model(model, num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(num_images, (10,10))

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloaders['Validation']):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for j in range(images.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Actual: {} \n Prediction: {}'.format(dog_breeds[labels[j]], dog_breeds[preds[j]]))
                
                image = images.cpu().data[j].numpy().transpose((1, 2, 0))
                
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                image = std * image + mean
                image = np.clip(image, 0, 1)
                
                plt.imshow(image)
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# In[20]:


# Make sure to comment this out when you go to "Commit" the kaggle notebook!
# otherwise, it'll run this model along with your other models down below.
model = train(model, criterion, optimizer, epochs, dataloaders, device)


# In[21]:


torch.save({
    'model' : CNN(),
    'epoch' : epochs,
    'model_state_dict': model.state_dict(),
    'optimizer' : optimizer,
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion' : criterion,
    'device' : device
}, 'base_model.pt')


# In[22]:


def load_checkpoint(filepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = checkpoint['criterion']
    epoch = checkpoint['epoch']
    model.to(device)

    return model, optimizer, criterion, epoch


# In[23]:


model, optimizer, criterion, epoch = load_checkpoint('base_model.pt')


# In[24]:


test_model(model, 6)


# In[25]:


class PreTrained_Resnet(nn.Module):
    def __init__(self):
        super(PreTrained_Resnet, self).__init__()
        
        # Loading up a pretrained ResNet18 model
        resnet = resnet18(pretrained = True)
        
        # Freeze the entire pretrained network
        for layer in resnet.parameters():
            layer.requires_grad = False
            
        self.feature_extraction = resnet
        
        # Write the classifier block for this network      
            # Tip: ResNet18's feature extraction portion ends up with 1000
            #   feature maps, and then implements a Global Average Pooling layer
            # So what would the size and dimension of the output tensor be?
            # Think about how can we take that output tensor and transform it
            #   into an array of dog breed predictions...
        self.classifier = nn.Sequential(
            # YOUR CODE HERE
            raise NotImplementedError()
        )
    
    # Write the forward method for this network (it's quite simple since we've
    #   defined the network in blocks already)
    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError()


# In[26]:


# Instantiate a pretrained network using the class we've just defined (call it
#  'pretrained')

# YOUR CODE HERE
raise NotImplementedError()

# Then define the loss function and optimizer to use for training (let's use
#   Adam again, with the same parameters as before)
# YOUR CODE HERE
raise NotImplementedError()

# Define your number of epochs to train and map your model to the gpu
# Keep epochs to 5 for time purposes during the workshop
# YOUR CODE HERE
raise NotImplementedError()

summary(pretrained, (3,224,224))


# In[27]:


pretrained = train(
    pretrained,
    criterion2,
    optimizer2,
    epochs2,
    dataloaders,
    device
)


# In[28]:


torch.save({
    'model' : PreTrained_Resnet(),
    'epoch' : epochs2,
    'model_state_dict': pretrained.state_dict(),
    'optimizer' : optimizer2,
    'optimizer_state_dict': optimizer2.state_dict(),
    'criterion' : criterion2,
    'device' : device
}, 'pretrained.pt')


# In[29]:


pretrained, optimizer2, criterion2, epoch2 = load_checkpoint('pretrained.pt')


# In[30]:


test_model(pretrained, 6)


# In[31]:


# Run this to generate the submission file for the competition!
### Make sure to name your model variable "pretrained" ###

# generate predictions
preds = []
pretrained = pretrained.to(device)
pretrained.eval()
for img in test_loader:
    outputs = pretrained(img.to(device))
    _, outputs = torch.max(outputs, 1)
    preds += [outputs.item()]

# create our pandas dataframe for our submission file. Squeeze removes
#   dimensions of 1 in a numpy matrix Ex: (161, 1) -> (161,)
indicies = ["{}.jpg".format(x) for x in range(len(image_datasets['Test']))]
preds = pd.DataFrame({'Id': indicies, 'Class': np.squeeze(preds)})

# save submission csv
preds.to_csv('submission.csv', header=['Id', 'Class'], index=False)
print("Submission generated!")

