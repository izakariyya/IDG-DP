import cv2
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
import random
from torchvision import models

import matplotlib.pyplot as plt
import numpy as np

#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

torch.manual_seed(123)
np.random.seed(123)

class HAR_Dataset(Dataset):
    """Our Human Activity Radar dataset"""
    def __init__(self, root_dir, activity=[], subject=[], repetition=[], transform=None):
        """
        Args:
		    root_dir (filepath): Directory with the radar data.
        activity (list): List of activities to include.
        subject (list): List of subjects to include.
        repetition (list): List of repetitions to include.
		    transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir=root_dir
        self.transform=transform

        file_names=[f[:-4] for f in os.listdir(root_dir) if f.endswith('.npy')]

        # filter samples based on given activity, subject and repetition
        samples=[]
        for f in file_names:
          if (not activity or int(f[0]) in activity) and (not subject or int(f[2:4]) in subject) and (not repetition or int(f[8]) in repetition):
            samples.append(f)
        self.sample_names = samples
        self.activities = activity
        self.subjects = subject

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        sample_id=self.sample_names[idx]
        mD_path=os.path.join(self.root_dir,sample_id+".npy")
        mD=np.load(mD_path)
        if mD.shape!=(800, 481):
          mD=cv2.resize(mD, (481, 800), interpolation=cv2.INTER_AREA) # shrinking array using inter area interpolation
        subject=self.subjects.index(int(sample_id[2:4]))
        activity=self.activities.index(int(sample_id[0]))
        repetition=sample_id[-1]

        sample = {'id':sample_id, 'mD':mD, 'subject':subject, 'activity':activity, 'repetition':repetition}

        if self.transform:
            sample=self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        sample_id, mD, subject, activity, repetition = sample['id'], sample['mD'], sample['subject'], sample['activity'], sample['repetition']
        return {'id': sample_id,
                'mD': torch.from_numpy(mD).float(),
                'subject': int(subject),
                'activity': int(activity),
                'repetition': int(repetition)}

class SliceUp(object):
    def __init__(self, window_size=100, time_shift=10):
        self.window_size = window_size
        self.time_shift = time_shift

    def __call__(self, sample):
        sample_id, mD, subject, activity, repetition = sample['id'], sample['mD'], sample['subject'], sample['activity'], sample['repetition']

        window_size_frames = int(self.window_size)
        time_shift_frames = int(self.time_shift)

        # Initialize the list to store sliced samples
        sliced_samples = []

        for start_frame in range(0, mD.shape[1] - window_size_frames + 1, time_shift_frames):
            end_frame = start_frame + window_size_frames
            sliced_mD = mD[:, start_frame:end_frame]

            sliced_sample = {
                'id': f"{sample_id}_slice_{start_frame}_{end_frame}",
                'mD': sliced_mD,
                'subject': subject,
                'activity': activity,
                'repetition': repetition
            }

            sliced_samples.append(sliced_sample)

        return sliced_samples

class RandomSlice(object):
     def __init__(self, window_size=300, probability=0.5):
        self.window_size = window_size
        self.probability = probability

     def __call__(self, sample):
        sample_id, mD, subject, activity, repetition = sample['id'], sample['mD'], sample['subject'], sample['activity'], sample['repetition']

        if random.random() < self.probability:
          window_size_frames = int(self.window_size)

          # Choose a random starting frame within the valid range
          start_frame = random.randint(0, mD.shape[1] - window_size_frames)

          end_frame = start_frame + window_size_frames
          sliced_mD = mD[:, start_frame:end_frame]

          if sliced_mD.shape!=(800, 481):
            sliced_mD=cv2.resize(sliced_mD, (481, 800), interpolation=cv2.INTER_AREA) # shrinking array using inter area interpolation

          sliced_sample = {
            'id': f"{sample_id}_random_slice_{start_frame}_{end_frame}",
            'mD': sliced_mD,
            'subject': subject,
            'activity': activity,
            'repetition': repetition
          }
          return sliced_sample
        else:
          return sample
    
DATASET = '019 West Cumbria Dataset'
train_dataset = HAR_Dataset(root_dir=DATASET, activity=[1,4,5], subject=[39,40,41,44,47,49,50,51,53,56], repetition=[1,2],
                            transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)


test_dataset = HAR_Dataset(root_dir=DATASET, activity=[1,4,5], subject=[39,40,41,44,47,49,50,51,53,56], repetition=[3],
                           transform=transforms.Compose([ToTensor()]))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

#Model
class MultiTaskResNet(nn.Module):
  def __init__(self, num_classes_subject, num_classes_activity, return_type='both'):
        super(MultiTaskResNet, self).__init__()

        # Load a pre-trained ResNet50 model
        resnet18 = models.resnet18()
        layer4 = resnet18.layer4

        # Remove first layer and the classification head (fc) and conv 5 of the original ResNet18 (layer4)
        self.resnet_features = nn.Sequential(*list(resnet18.children())[1:-3])

        self.first_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Create new branches - layer 4 sequential with the weights and new fc for each.
        self.subject_branch = nn.Sequential(layer4, resnet18.avgpool, nn.Flatten(), nn.Linear(512, num_classes_subject))
        self.activity_branch = nn.Sequential(layer4, resnet18.avgpool, nn.Flatten(), nn.Linear(512, num_classes_activity))

        self.return_type = return_type
  def forward(self, x):
        # Extract features using ResNet backbone
        x = self.first_layer(x)
        features = self.resnet_features(x)

        # Separate convolutional blocks for subject and activity branches
        subject = self.subject_branch(features)
        activity = self.activity_branch(features)

        if self.return_type == 'subject':
          return subject
        if self.return_type == 'activity':
          return activity
        else:
          return subject, activity
        
#Model with noise
class MultiTaskResNetNoise(nn.Module):
    def __init__(self, num_classes_subject, num_classes_activity, return_type='both', noise_indices=None, epsilon=0.2):
        super(MultiTaskResNetNoise, self).__init__()

        # Load a pre-trained ResNet18 model
        resnet18 = models.resnet18()
        layer4 = resnet18.layer4

        # Remove the first layer and the classification head (fc) and conv 5 of the original ResNet18 (layer4)
        self.resnet_features = nn.Sequential(*list(resnet18.children())[1:-3])

        self.first_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Create new branches - layer 4 sequential with the weights and new fc for each.
        self.subject_branch = nn.Sequential(layer4, resnet18.avgpool, nn.Flatten(), nn.Linear(512, num_classes_subject))
        self.activity_branch = nn.Sequential(layer4, resnet18.avgpool, nn.Flatten(), nn.Linear(512, num_classes_activity))

        self.return_type = return_type
        self.noise_indices = noise_indices
        self.epsilon = epsilon

    def forward(self, x):
        # Extract features using ResNet backbone
        # Add random noise to features at specified indices
        if self.noise_indices is not None:
            laplace_noise = np.random.laplace(0, 1.0/self.epsilon, x.shape)
            laplace_noise = torch.from_numpy(laplace_noise.astype('float32')).to(x.device)
            x[:, :, self.noise_indices[0].to(device), self.noise_indices[1].to(device)] += laplace_noise[:, :, self.noise_indices[0].to(device), self.noise_indices[1].to(device)]

        x = self.first_layer(x)
        features = self.resnet_features(x)


        # Separate convolutional blocks for subject and activity branches
        subject = self.subject_branch(features)
        activity = self.activity_branch(features)

        if self.return_type == 'subject':
            return subject
        if self.return_type == 'activity':
            return activity
        else:
            return subject, activity
        

# modified EarlyStopper from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, model, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model = model

    def early_stop(self, validation_loss, current_model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = current_model
            print('[New best!] {:.3f}'.format(self.min_validation_loss))
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model, optimizer, criterion1, train_dataloader):
  model.train()
  running_loss = 0.0
  for i, data in enumerate(train_dataloader, 0):
    inputs = data['mD'].to(device)
    activity_labels = data['activity'].to(device)
    inputs = inputs.unsqueeze(1)  # Add channel dimension

    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      activity_output  = model(inputs)
      loss = criterion1(activity_output, activity_labels)
      loss.backward()
      optimizer.step()
    running_loss += loss.item()
  train_loss =  running_loss/len(train_dataloader)
  return model, train_loss

def validate(model, criterion1, test_dataloader):
  model.eval()
  running_loss = 0.0
  for i, data in enumerate(test_dataloader, 0):
    inputs = data['mD'].to(device)
    subject_labels = data['subject'].to(device)
    activity_labels = data['activity'].to(device)
    inputs = inputs.unsqueeze(1)  # Add channel dimension
    activity_output = model(inputs)
    loss = criterion1(activity_output, activity_labels)
    running_loss += loss.item()
  val_loss = running_loss/len(test_dataloader)
  return val_loss

def train_subject(model, activity_features_model, optimizer, criterion1, train_dataloader):
  model.train()
  running_loss = 0.0
  for i, data in enumerate(train_dataloader, 0):
    inputs = data['mD'].to(device)
    subject_labels = data['subject'].to(device)
    inputs = inputs.unsqueeze(1)  # Add channel dimension
    features = activity_features_model(inputs)

    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      output  = model(features)
      loss = criterion1(output, subject_labels)
      loss.backward()
      optimizer.step()
    running_loss += loss.item()
  train_loss =  running_loss/len(train_dataloader)
  return model, train_loss

def validate_subject(model, activity_features_model, criterion1, test_dataloader):
  model.eval()
  running_loss = 0.0
  for i, data in enumerate(test_dataloader, 0):
    inputs = data['mD'].to(device)
    subject_labels = data['subject'].to(device)
    inputs = inputs.unsqueeze(1)  # Add channel dimension
    features = activity_features_model(inputs)
    output = model(features)
    loss = criterion1(output, subject_labels)
    running_loss += loss.item()
  val_loss = running_loss/len(test_dataloader)
  return val_loss

def plot_cm(true, pred, num, title):
  confusion_matrix = metrics.confusion_matrix(true, pred)
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = np.arange(num))
  cm_display.plot()
  plt.title(title)
  plt.show()

  accuracy = metrics.accuracy_score(true, pred)
  precision = metrics.precision_score(true, pred, average='micro')
  recall = metrics.recall_score(true, pred, average='micro')
  f1_score = metrics.f1_score(true, pred, average='macro')

  print("Accuracy {:.3}, Recall {:.3}, Precission {:.3}, F1-score {:.3}".format(accuracy, recall, precision, f1_score))

def evaluate_model(model, test_loader, activity_num):
  true_labels_a = []
  pred_labels_a = []

  model.eval()
  with torch.no_grad():
     for data in test_loader:
      inputs = data['mD'].to(device)
      subject_labels = data['subject'].to(device)
      activity_labels = data['activity'].to(device)
      inputs = inputs.unsqueeze(1)  # Add channel dimension
      activity_output = model(inputs)
      _, predicted_activity = torch.max(activity_output, 1)

      true_labels_a.extend(activity_labels.cpu().tolist())
      pred_labels_a.extend(predicted_activity.cpu().tolist())
  plot_cm(true_labels_a, pred_labels_a, activity_num, "Activity Recognition MultiTask Model ResNet18")

def evaluate_model_subject(model, activity_features_model, test_loader, subject_num):
  true_labels_s = []
  pred_labels_s = []

  model.eval()
  with torch.no_grad():
     for data in test_loader:
      inputs = data['mD'].to(device)
      subject_labels = data['subject'].to(device)
      inputs = inputs.unsqueeze(1)  # Add channel dimension
      features = activity_features_model(inputs)
      subject_output = model(features)
      _, predicted_subject = torch.max(subject_output, 1)

      true_labels_s.extend(subject_labels.cpu().tolist())
      pred_labels_s.extend(predicted_subject.cpu().tolist())
  plot_cm(true_labels_s, pred_labels_s, subject_num, "Subject Recognition MultiTask Model ResNet18")


attributions_a = np.load('saliency_attributions_activity_10.npy')
attributions_s = np.load('saliency_attributions_subject_10.npy')

average_as = np.mean(abs(np.array(attributions_a))-abs(np.array(attributions_s)), axis=0)
std_dev_as = np.std(abs(np.array(attributions_a))-abs(np.array(attributions_s)), axis=0)

# Flatten the nested list to a 1D array
flat_values = np.array(average_as).flatten() * 1000

# Separate positive and negative values
positive_values = flat_values[flat_values >= 0]
negative_values = flat_values[flat_values < 0]

# Create histograms for positive and negative values
plt.hist(positive_values, bins=50, color='green', edgecolor='black', alpha=0.7, label='Activity')
plt.hist(negative_values, bins=50, color='red', edgecolor='black', alpha=0.7, label='Subject')

# Add labels and title
plt.xlabel('Node Attribution Value')
plt.ylabel('Frequency')
plt.title('Histogram of (Activity-Subject)*1000 Node Attribution Values')

# Add legend
plt.legend()

# Show the plot
plt.show()

attribution_threshold = 0
noise_indices=np.where(np.array(average_as).squeeze() < attribution_threshold)
noise_indices=torch.tensor(np.array(noise_indices)).to(device)
epsilon = 1.2

num_classes_activity = 3  # number of activities

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Create an instance of the MultiTaskResNet model
activity_recognition = MultiTaskResNetNoise(10, num_classes_activity, return_type='activity', noise_indices=noise_indices, epsilon=epsilon)

activity_recognition = activity_recognition.to(device)
criterion1 = nn.CrossEntropyLoss()
optimizer = optim.SGD(activity_recognition.parameters(), lr=0.001, momentum=0.9)
n_epochs = 400
early_stopper = EarlyStopper(activity_recognition, patience=8, min_delta=0.01)
train_losses = []
validation_losses = []

for epoch in np.arange(n_epochs):
    activity_recognition, train_loss = train(activity_recognition, optimizer, criterion1, train_loader)
    validation_loss = validate(activity_recognition, criterion1, test_loader)
    train_losses += [train_loss]
    validation_losses += [validation_loss]
    #print(f'[Epoch: {epoch + 1}] Train loss: {train_loss:.3f} Val loss: {validation_loss:.3f}')
    if (early_stopper.early_stop(validation_loss, activity_recognition) and epoch > 200):
      activity_recognition = early_stopper.best_model
      break

epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, validation_losses, label='Validation Loss')

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Losses')

# Adding a legend
plt.legend()

# Display the plot
plt.show()


evaluate_model(activity_recognition, test_loader, num_classes_activity)

torch.save(activity_recognition.state_dict(), 'noise_activity_model_10.pt')