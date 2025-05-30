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

#Dataset

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



#Five Subject Selection
#DATASET = '019 West Cumbria Dataset'
#train_dataset = HAR_Dataset(root_dir=DATASET, activity=[1,4,5], subject=[39,40,41,44,56], repetition=[1,2],
#                           transform=transforms.Compose([ToTensor()]))
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)

#test_dataset = HAR_Dataset(root_dir=DATASET, activity=[1,4,5], subject=[39,40,41,44,56], repetition=[3],
#                           transform=transforms.Compose([ToTensor()]))
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

#10 Subject Selection
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

#Baseline
# LOAD PRETRAINED MODEL
PATH = 'multi_task_10.pt'
model = torch.load(PATH)

model = MultiTaskResNet(10, 3)
model.load_state_dict(torch.load(PATH))
model.to(device)
model.eval()

## SHOW PREDICTIONS FOR 10 SUBJECTS
dataiter = iter(test_loader)
data = next(dataiter)
inputs_id = data['id']
inputs = data['mD'].to(device)
subject_labels = data['subject'].to(device)
activity_labels = data['activity'].to(device)
inputs = inputs.unsqueeze(1)

subject_output, activity_output = model(inputs)
_, predicted_subject = torch.max(subject_output, 1)
_, predicted_activity = torch.max(activity_output, 1)

print('Activity Label:     ', activity_labels.cpu().tolist())
print('Activity Predicted: ', predicted_activity.cpu().tolist())


print('\nSubject Label:      ', subject_labels.cpu().tolist())
print('Subject Predicted:  ', predicted_subject.cpu().tolist())

# Calculate accuracy
activity_accuracy = torch.sum(predicted_activity == activity_labels).item() / len(activity_labels)
subject_accuracy = torch.sum(predicted_subject == subject_labels).item() / len(subject_labels)

# Print accuracy
print('Activity Accuracy: {:.2%}'.format(activity_accuracy))
print('Subject Accuracy: {:.2%}'.format(subject_accuracy))

#Captum
#PATH = 'multi_task_10.pt'
# Two models for activity/subject predictions
model_a = MultiTaskResNet(10, 3, return_type='activity')
model_a.load_state_dict(torch.load(PATH))
model_a.to(device)
model_a.eval()

model_s = MultiTaskResNet(10, 3, return_type='subject')
model_s.load_state_dict(torch.load(PATH))
model_s.to(device)
model_s.eval()

#Gradcam
def get_layer_gradcam(data, model_a, model_s, layer_a, layer_s):
  inputs = data['mD'].to(device)
  subject_labels = data['subject'].to(device)
  activity_labels = data['activity'].to(device)
  inputs = inputs.unsqueeze(1)  # Add channel dimension

  layer_gradcam_a = LayerGradCam(model_a, layer_a)
  attributions_lgc_a = layer_gradcam_a.attribute(inputs, target=activity_labels)

  layer_gradcam_s = LayerGradCam(model_s, layer_s)
  attributions_lgc_s = layer_gradcam_s.attribute(inputs, target=subject_labels)

  return attributions_lgc_a, attributions_lgc_s

## each sample compute layer attribution and store in x
layer_attributions_a = []
layer_attributions_s = []

with torch.no_grad():
   for data in train_loader:
      inputs = data['mD'].to(device)
      subject_labels = data['subject'].to(device)
      activity_labels = data['activity'].to(device)
      inputs = inputs.unsqueeze(1)  # Add channel dimension

      att_a , att_s = get_layer_gradcam(data, model_a, model_s, model_a.resnet_features[5][1].conv2, model_s.resnet_features[5][1].conv2)

      subject_output, activity_output = model(inputs)
      _, predicted_subject = torch.max(subject_output, 1)
      _, predicted_activity = torch.max(activity_output, 1)

      # interpolate
      upsamp_attr_lgc_a = LayerAttribution.interpolate(att_a, inputs.shape[2:])
      upsamp_attr_lgc_s = LayerAttribution.interpolate(att_s, inputs.shape[2:])

      layer_attributions_a.extend(upsamp_attr_lgc_a.cpu().tolist())
      layer_attributions_s.extend(upsamp_attr_lgc_s.cpu().tolist())

#Get outputted Graph
average_a = np.mean(layer_attributions_a, axis=0)
std_dev_a = np.std(layer_attributions_a, axis=0)

#_ = viz.visualize_image_attr(np.transpose(abs(average_a), axes=(1,2,0)),
#                             sign="positive",
#                             title="",
#                             show_colorbar=True)

#_ = viz.visualize_image_attr(np.transpose(std_dev_a, axes=(1,2,0)),
#                             sign="all",
#                             title="standard deviation",
#                             show_colorbar=True)

#average_s = np.mean(layer_attributions_s, axis=0)
#std_dev_s = np.std(layer_attributions_s, axis=0)

#_ = viz.visualize_image_attr(np.transpose(-abs(average_s), axes=(1,2,0)),
#                             sign="negative",
#                             title="",
#                             show_colorbar=True)

#_ = viz.visualize_image_attr(np.transpose(std_dev_s, axes=(1,2,0)),
#                             sign="all",
#                             title="standard deviation",
#                             show_colorbar=True)

average_as = np.mean(abs(np.array(layer_attributions_a))-abs(np.array(layer_attributions_s)), axis=0)
std_dev_as = np.std(abs(np.array(layer_attributions_a))-abs(np.array(layer_attributions_s)), axis=0)

#_ = viz.visualize_image_attr(np.transpose(average_as, axes=(1,2,0)),
#                             sign="all",
#                             title="A-S Average Layer Attribution, last shared conv",
#                             show_colorbar=True)

#_ = viz.visualize_image_attr(np.transpose(std_dev_as, axes=(1,2,0)),
#                             sign="positive",
#                             title="A-S Average Layer Attribution, last shared conv",
#                             show_colorbar=True)

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
# plt.title('Histogram of (Activity-Subject)*1000 Node Attribution Values')

# Add legend
plt.legend()

# Show the plot
#plt.show()
plt.savefig('Results/Hist_10_Subject_Neg_pos')


np.save('gradcam_attributions_activity_10.npy', layer_attributions_a)
np.save('gradcam_attributions_subject_10.npy', layer_attributions_s)

#!cp gradcam_attributions_activity_10.npy /content/drive/MyDrive/gradcam_attributions_activity_10.npy
#!cp gradcam_attributions_subject_10.npy /content/drive/MyDrive/gradcam_attributions_subject_10.npy

attributions_a = np.load('gradcam_attributions_activity_10.npy')
attributions_s = np.load('gradcam_attributions_subject_10.npy')

average_as = np.mean(abs(np.array(attributions_a))-abs(np.array(attributions_s)), axis=0)
std_dev_as = np.std(abs(np.array(attributions_a))-abs(np.array(attributions_s)), axis=0)

average_s = np.mean(attributions_s, axis=0)
# norm_attr_s = viz._normalize_attr(np.transpose(average_s, axes=(1,2,0)), 'all', 2, reduction_axis=2)

average_a = np.mean(attributions_a, axis=0)
# norm_attr_a = viz._normalize_attr(np.transpose(average_a, axes=(1,2,0)), 'all', 2, reduction_axis=2)


average_as = average_a - average_s

#_ = viz.visualize_image_attr(np.transpose(average_as, axes=(1,2,0)),
#                             sign="all",
#                             title="A-S Average Layer Attribution, last shared conv",
#                             show_colorbar=True)

average_as = viz._normalize_attr(np.transpose(average_as, axes=(1,2,0)), 'all', 2, reduction_axis=2)

# Flatten the nested list to a 1D array
flat_values = np.array(average_as).flatten()

# Separate positive and negative values
positive_values = flat_values[flat_values >= 0]
negative_values = flat_values[flat_values < 0]

# Create histograms for positive and negative values
plt.hist(positive_values, bins=60, color='green', edgecolor='black', alpha=0.7, label='Activity')
plt.hist(negative_values, bins=60, color='red', edgecolor='black', alpha=0.7, label='Subject')

# Add labels and title
plt.xlabel('Pixel Attribution Level')
plt.ylabel('Frequency')

# Add legend
plt.legend()

# Show the plot
#plt.show()
plt.savefig('Results/Hist_neg_pos_val_second')

#Adding Noise Gradcam Attribute

x_values = np.geomspace(2.5, 0.05, num=15)


results_a = []
results_s = []
intervals = np.linspace(-1.0,0,15)
for attribution_threshold in intervals:
  noise_indices=np.where(np.array(average_as).squeeze() < attribution_threshold)
  noise_indices=torch.tensor(np.array(noise_indices)).to(device)

  a_accs = []
  s_accs = []

  for x in np.geomspace(2.5, 0.05, num=15):
    noise_model = MultiTaskResNetNoise(10, 3, noise_indices=noise_indices, epsilon=x)
    noise_model.load_state_dict(torch.load(PATH))
    noise_model.to(device)
    noise_model.eval()

    dataiter = iter(test_loader)
    data = next(dataiter)
    inputs_id = data['id']
    inputs = data['mD'].to(device)
    subject_labels = data['subject'].to(device)
    activity_labels = data['activity'].to(device)
    inputs = inputs.unsqueeze(1)

    subject_output, activity_output = noise_model(inputs)
    _, predicted_subject = torch.max(subject_output, 1)
    _, predicted_activity = torch.max(activity_output, 1)

    # Calculate f-1 score
    f1_subject = f1_score(subject_labels.cpu(), predicted_subject.cpu(), average='macro')
    f1_activity = f1_score(activity_labels.cpu(), predicted_activity.cpu(), average='macro')

    a_accs.append(f1_activity)
    s_accs.append(f1_subject)

    # Plotting
  plt.figure(figsize=(10, 6))
  plt.plot(x_values, a_accs, label='Activity')
  plt.plot(x_values, s_accs, label='Subject')
  plt.xlabel('Epsilon')
  plt.xlim(2.5, 0)
  plt.ylabel('F-1 Score')
  plt.title('Activity and Subject F1 score for attribution threshold ' +str(attribution_threshold))
  plt.legend()
  #plt.show()
  plt.savefig('Results/activity_subject_attribution_epsilon')

  results_a.append(a_accs)
  results_s.append(s_accs)

