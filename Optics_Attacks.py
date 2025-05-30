from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
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

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#torch.manual_seed(123)
#np.random.seed(123)


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
    

#Five Subject Selection

DATASET = '019 West Cumbria Dataset'

#train_dataset_5mm = HAR_Dataset(root_dir=DATASET, activity=[1,4,5], subject=[39,40,41,44,56], repetition=[1,2],
#                           transform=transforms.Compose([ToTensor()]))
#train_loader_5mm = DataLoader(train_dataset_5mm, batch_size=32, shuffle=False, num_workers=2)

#test_dataset_5mm = HAR_Dataset(root_dir=DATASET, activity=[1,4,5], subject=[39,40,41,44,56], repetition=[3],
#                           transform=transforms.Compose([ToTensor()]))
#test_loader_5mm = DataLoader(test_dataset_5mm, batch_size=32, shuffle=False, num_workers=2)

#10 Subject Selection

#DATASET = '019 West Cumbria Dataset'
train_dataset_O10 = HAR_Dataset(root_dir=DATASET, activity=[1,4,5], subject=[39,40,41,44,47,49,50,51,53,56], repetition=[1,2],
                            transform=transforms.Compose([ToTensor()]))
train_loader_O10 = DataLoader(train_dataset_O10, batch_size=32, shuffle=False, num_workers=2)


test_dataset_O10 = HAR_Dataset(root_dir=DATASET, activity=[1,4,5], subject=[39,40,41,44,47,49,50,51,53,56], repetition=[3],
                           transform=transforms.Compose([ToTensor()]))
test_loader_O10 = DataLoader(test_dataset_O10, batch_size=32, shuffle=False, num_workers=2)


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


def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

#Membership Inference Attack

def black_box_attack(target_model, x_train, y_train, x_test, y_test, attack_train_size, attack_test_size):
  mlp_attack_bb = MembershipInferenceBlackBox(target_model, attack_model_type='nn')

  # train attack model
  mlp_attack_bb.fit(x_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size],
              x_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size])

  # infer
  mlp_inferred_train_bb = mlp_attack_bb.infer(x_train[attack_train_size:].astype(np.float32), y_train[attack_train_size:])
  mlp_inferred_test_bb = mlp_attack_bb.infer(x_test[attack_test_size:].astype(np.float32), y_test[attack_test_size:])

  # check accuracy
  mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
  mlp_test_acc_bb = 1 - ((np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb)))
  mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))

  print(f"Members Accuracy: {mlp_train_acc_bb:.4f}")
  print(f"Non Members Accuracy {mlp_test_acc_bb:.4f}")
  print(f"Attack Accuracy {mlp_acc_bb:.4f}")

  precision, recall = calc_precision_recall(np.concatenate((mlp_inferred_train_bb, mlp_inferred_test_bb)),
                              np.concatenate((np.ones(len(mlp_inferred_train_bb)), np.zeros(len(mlp_inferred_test_bb)))))
  print(f"Attack Precision {precision:.4f}")
  print(f"Attack Recall {recall:.4f}")
  

  #Five Subject

#x_train_5 = np.array([x['mD'] for x in train_dataset_5mm])
#y_train_5 = np.array([y['activity'] for y in train_dataset_5mm])
#x_test_5 = np.array([x['mD'] for x in test_dataset_5mm])
#y_test_5 = np.array([y['activity'] for y in test_dataset_5mm])

#x_test_5 = np.expand_dims(x_test_5, axis=1)
#x_train_5 = np.expand_dims(x_train_5, axis=1)

#attack_train_ratio = 0.5
#attack_train_size_5 = int(len(x_train_5) * attack_train_ratio)
#attack_test_size_5 = int(len(x_test_5) * attack_train_ratio)


x_train_10n = np.array([x['mD'] for x in train_dataset_O10])
y_train_10n = np.array([y['activity'] for y in train_dataset_O10])
x_test_10n = np.array([x['mD'] for x in test_dataset_O10])
y_test_10n = np.array([y['activity'] for y in test_dataset_O10])

x_test_10 = np.expand_dims(x_test_10n, axis=1)
x_train_10 = np.expand_dims(x_train_10n, axis=1)

attack_train_ratio = 0.5
attack_train_size_10 = int(len(x_train_10) * attack_train_ratio)
attack_test_size_10 = int(len(x_test_10n) * attack_train_ratio)


from art.estimators.classification.pytorch import PyTorchClassifier

PATHNNM = 'mask_model_10_new.pt'

model10MN = MultiTaskResNet(10, 3, return_type='activity')
model10MN.load_state_dict(torch.load(PATHNNM, map_location=torch.device('cpu')))
model10MN.eval()

mlp_art_model_10MN = PyTorchClassifier(model=model10MN, loss=nn.CrossEntropyLoss(), input_shape=(1,800,481), nb_classes=3)

test_pred = np.array([np.argmax(arr) for arr in mlp_art_model_10MN.predict(x_test_10.astype(np.float32))])
print('Noise model Five Test accuracy: ', np.sum(test_pred == y_test_10n) / len(y_test_10n))

black_box_attack(mlp_art_model_10MN, x_train_10, y_train_10n, x_test_10, y_test_10n, attack_train_size_10, attack_test_size_10)

#Noise Target Model

#PATH10N = 'Latest_Gradcam_noise_activity_model_10.pt'

#model10N = MultiTaskResNet(10, 3, return_type='activity')
#model10N.load_state_dict(torch.load(PATH10N, map_location=torch.device('cpu')))
#model10N.eval()

#mlp_art_model_noise_10 = PyTorchClassifier(model=model10N, loss=nn.CrossEntropyLoss(), input_shape=(1,800,481), nb_classes=3)

#test_pred = np.array([np.argmax(arr) for arr in mlp_art_model_noise_10.predict(x_test_10.astype(np.float32))])
#print('Noise model Ten Test accuracy: ', np.sum(test_pred == y_test_10) / len(y_test_10))
#black_box_attack(mlp_art_model_noise_10, x_train_10, y_train_10, x_test_10, y_test_10, attack_train_size_10, attack_test_size_10)