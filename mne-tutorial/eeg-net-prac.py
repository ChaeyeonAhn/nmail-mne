import warnings
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import torch.nn.functional as F
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.svm import SVC
from scipy.signal import iirnotch, filtfilt
from mne.io import concatenate_raws, read_raw_edf
import json
import matplotlib.pyplot as plt
import torch.nn as nn
import os
warnings.filterwarnings('ignore')
DATASET_ROOT = "C:/Users/NMAIL/nmail-mne/mne-tutorial/"


# MI, 250Hz, C3,Cz,C4: 7,9,11 
# l/r hand: 0,1 foot, tongue: 2,3 LABELS
# subj: 1-9, 0.5-100Hz, 10-20 system, returns 7.5s data, 2s before and 1.5s after the trial
def get_data_2a(subject, training, root_path=DATASET_ROOT+'eeg-net-data/'):
    '''	Loads the dataset 2a of the BCI Competition IV
    available on http://bnci-horizon-2020.eu/database/data-sets

    Keyword arguments:
    subject -- number of subject in [1, .. ,9]
    training -- if True, load training data
                if False, load testing data

    Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1825 channel * timepoint
            class_return 	numpy matrix 	size = NO_valid_trial
    '''
    # two sessions per subject, trial: 4s, 

    # reference: left mastoid, ground: right mastoid
    # sampling rate: 250Hz
    # bandpassed: 0.5-100Hz
    NO_channels = 22
    NO_tests = 6 * 48 

    # Each session is comprised of 6 runs separated by short breaks. One run consists of 48 trials
    Window_Length = int(7.5 * 250) # = 1875

    # 미리 최종 데이터 집어넣을 컨테이너 마련 (numpy array)
    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests, NO_channels, Window_Length))

    NO_valid_trial = 0 # "유효한 시도 수"

    # 모드에 따라 원하는 데이터를 불러와서 'data' 키에 해당하는 데이터를 가져온다.
    if training:
        a = sio.loadmat(root_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(root_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    # print(a_data)

    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0] # entire data (refer to comment before for loop) EEG data
        a_trial = a_data3[1] # trials
        a_y = a_data3[2] # labels (MI)
        a_fs = a_data3[3] # sampling freq
        a_classes = a_data3[4] # our class list len = 4
        a_artifacts = a_data3[5] # noise 0 or 1 generally
        a_gender = a_data3[6] 
        a_age = a_data3[7]
        # a_trial.size > 0 means there is data in this run
        for trial in range(0, a_trial.size):
            # remove bad trials which have artifacts
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial]) + Window_Length), :NO_channels]) # 4s pre-trial and 4s post-trial
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    # index 500~1500 is the imagery time
    data = data_return[0:NO_valid_trial, :, :]
    class_return = class_return[0:NO_valid_trial]-1 # 1, 2, 3, 4 -> 0, 1, 2, 3

    return np.array(data), np.array(class_return)

data1, class_return1 = get_data_2a(1, True)
data2, class_return2 = get_data_2a(2, True)
data3, class_return3 = get_data_2a(3, True)
data4, class_return4 = get_data_2a(4, True)
data5, class_return5 = get_data_2a(5, True)
# data6, class_return6 = get_data_2a(6, True)
# data7, class_return7 = get_data_2a(7, True)
# data8, class_return8 = get_data_2a(8, True)
# data9, class_return9 = get_data_2a(9, True)

data = data5
class_return = class_return5
data = data[:, :, 500:1500]

data_test1, class_test1 = get_data_2a(1, False)
data_test2, class_test2 = get_data_2a(2, False)
data_test3, class_test3 = get_data_2a(3, False)
data_test4, class_test4 = get_data_2a(4, False)
data_test5, class_test5 = get_data_2a(5, False)
# data_test6, class_test6 = get_data_2a(6, False)
# data_test7, class_test7 = get_data_2a(7, False)
# data_test8, class_test8 = get_data_2a(8, False)
# data_test9, class_test9 = get_data_2a(9, False)
data_test = data_test5
class_test = class_test5
data_test = data_test[:, :, 500:1500]


def EEG_array_modifier(eeg_array, label_array):
    events_array = np.column_stack((np.arange(len(label_array)), np.zeros(len(label_array), dtype=int), label_array)).astype(int)
    # 3 개의 1D array를 세로로 쌓아서 event array 생성
    return np.array(eeg_array), events_array


def EEG_to_epochs(eeg_array, label_array, sfreq = 250, event_id = {'LeftHand': 0, 'RightHand': 1, 'Feet': 2, 'Tongue': 3}):
    channels = ['Fz', 'FC3', 'C5', 'CP3', 'P1', 'POz', 'P2', 'CP4', 'C6', 'FC4', 'FC1', 'C3', 'CP1', 'Pz', 'CP2', 'C4', 'FC2', 'FCz', 'C1', 'CPz', 'C2', 'Cz']
    n_channels = 22
    ch_types = ['eeg'] * 22
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)
    info = info.set_montage(montage)
    data, events = EEG_array_modifier(eeg_array, label_array)
    epochs = mne.EpochsArray(data, info, np.array(events), tmin=0, event_id=event_id)
    return epochs, events

epochs, events = EEG_to_epochs(data, class_return)
epochs_test, events_test = EEG_to_epochs(data_test, class_test)

################################################
# Band Pass filtering (0.5 - 100)
epochs = epochs.filter(l_freq=0.5, h_freq=100)
epochs_test = epochs_test.filter(l_freq=0.5, h_freq=100)
# fig = epochs.compute_psd(fmax=125, method="welch", n_fft=1000).plot(
#     average=True, amplitude=False, exclude="bads"
# )
# plt.show()

def apply_notch_filter(data, sfreq, freq=60.0, quality_factor=30.0):

    # 노치 필터의 주파수와 품질 인자 설정
    # 60Hz는 전원선 노이즈
    b, a = iirnotch(w0=freq, Q=quality_factor, fs=sfreq)
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

data = apply_notch_filter(epochs.get_data(), 250, 50, 30)
data = apply_notch_filter(data, 250, 100, 30)

data_test = apply_notch_filter(epochs_test.get_data(), 250, 50, 30)
data_test = apply_notch_filter(data_test, 250, 100, 30)


def standardize_data(data):
    n_epochs, n_channels, n_timepoints = data.shape
    standardized_data = np.zeros((n_epochs, n_channels, n_timepoints))
    scaler = StandardScaler()
    for channel in range(n_channels): # 채널별로 다 표준화
        scaled_data = scaler.fit_transform(data[:, channel, :].T).T
        standardized_data[:, channel, :] = scaled_data
    return standardized_data

def normalize_data(data):
    n_epochs, n_channels, n_timepoints = data.shape
    normalized_data = np.zeros((n_epochs, n_channels, n_timepoints))
    scaler = MinMaxScaler()
    for channel in range(n_channels):
        scaled_data = scaler.fit_transform(data[:, channel, :].T).T
        normalized_data[:, channel, :] = scaled_data
    return normalized_data

##############################################
# Training Data stan + norm
data = standardize_data(data)
data = normalize_data(data)
# Testing Data stan + norm
data_test = standardize_data(data_test)
data_test = normalize_data(data_test)

# plt.plot(data[0])
# data, label = EEG_to_epochs(data, class_return)
# fig = data.compute_psd(fmax=125, method="welch", n_fft=1000).plot(
#     average=True, amplitude=False, exclude="bads"
# )
# plt.show()
# data = data.get_data()
# data_test = data_test.get_data()

################################################
# Slicing data into small overlapping pieces
data0 = data[:, :, 0:900]
print(data0.shape)
data1 = data[:, :, 50:950]
print(data1.shape)
data2 = data[:, :, 100:1000]
print(data2.shape)

data_test0 = data_test[:, :, 0:900]
print(data_test0.shape)
data_test1 = data_test[:, :, 50:950]
print(data_test1.shape)
data_test2 = data_test[:, :, 100:1000]
print(data_test2.shape)

data = np.concatenate((data0, data1, data2))
data_test = np.concatenate((data_test0, data_test1, data_test2))
class_return = np.concatenate((class_return, class_return, class_return))
class_test = np.concatenate((class_test, class_test, class_test))

class EEGNet(nn.Module):
    def __init__(
            self,
            num_channels,  # number of channels
            F_1,
            F_2,
            D,
            dropout_prob=0.25 # or 0.5
    ):
        super(EEGNet, self).__init__()

        # self.last_size = last_size 
        self.num_channels = num_channels
        self.conv_temp = nn.Conv2d(1, F_1, kernel_size=(1, 64), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F_1, momentum=0.1, affine=True, eps=1e-5)
        self.depth_conv = nn.Conv2d(F_1, D*F_1, kernel_size=(num_channels, 1), bias=False, groups=F_1) 
        # group 수가 입력 채널 수랑 같으면 = depthwise convolution 가능하게 해주는 것
        self.batchnorm2 = nn.BatchNorm2d(D*F_1, momentum=0.1, affine=True, eps=1e-5)
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.sep_conv1 = nn.Conv2d(D*F_1, D*F_1, kernel_size=(1, 16), bias=False, groups=D*F_1) #sep_1 도 depthwise conv
        self.sep_conv2 = nn.Conv2d(D*F_1, F_2, kernel_size=(1, 1), bias=False) 
        self.batchnorm3 = nn.BatchNorm2d(F_2, momentum=0.1, affine=True, eps=1e-5)
        self.elu = nn.ELU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(384, 4) 

    def forward(self, input):
        if len(input.shape)==3:
            input = input.unsqueeze(1)
        # print("input: ", input.shape)
        x = self.conv_temp(input)
        # print("conv_temp: ",x.shape)

        x = self.batchnorm1(x) # 어차피 Conv 뒤에 Batch Norm을 쓰기 때문에 Conv에 bias는 소용 없으므로 False라고 한다.
        # print("batchnorm1: ",x.shape)
        x = self.depth_conv(x)
        # print("depth_conv: ",x.shape)
        x = self.batchnorm2(x)
        # print("batchnorm2: ",x.shape)
        x = self.elu(x)
        # x = torch.clamp(x, min=0, max=1)
        x = self.avgpool1(x)
        # x = torch.clamp(x, min=0, max=1)
        # print("avgpool1: ",x.shape)
        x = self.dropout(x)
        # x = torch.clamp(x, min=0, max=1)
        x = self.sep_conv1(x)
        # print("sep_conv1: ",x.shape)
        x = self.sep_conv2(x)
        # print("sep_conv2: ",x.shape)
        x = self.batchnorm3(x)
        # print("batchnorm3: ",x.shape)
        x = self.elu(x)
        # x = torch.clamp(x, min=0, max=1)
        x = self.avgpool2(x)
        # x = torch.clamp(x, min=0, max=1)
        # print("avgpool2: ",x.shape)
        x = self.dropout(x)
        # x = torch.clamp(x, min=0, max=1)
        x = self.flatten(x)
        # print("flatten: ",x.shape)
        output = self.fc(x)
        # print("linear: ",output.shape)
        return output

    
########################################################
# Making dataloader
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(class_return, dtype=torch.int64) # 얘는 넘파이 상태

data_t_tensor = torch.tensor(data_test, dtype=torch.float32)
labels_t_tensor = torch.tensor(class_test, dtype=torch.int64)

train_dataset = TensorDataset(data_tensor, labels_tensor)
test_dataset = TensorDataset(data_t_tensor, labels_t_tensor)

train_loader1 = DataLoader(train_dataset, batch_size=32, shuffle=True) # 훈련 시 쓸 데이터
eval_loader = DataLoader(test_dataset, batch_size=32, shuffle=True) # 검증 데이터

def train_model(model, train_loader, criterion, optimizer): 
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    return avg_loss

def evaluate_model(model, eval_loader, criterion):
    model.eval() # 이게 모델을 어떤 모드로 설정하느냐에 따라 모델이 조금씩 달라진다.
    # 특히 이렇게 검증하는 단계에서는 drop out을 안 한다고 했던 것 같다..!
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = test_loss / len(eval_loader) # 한 배치에서 손실들의 합 / 배치 개수
    accuracy = correct / total # 몇 개 맞췄는지 / 몇 개 예상했는지
    return avg_loss, accuracy

#############################################################################
# 폴드 5개 교차 검증 후 검증

kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_results = []
eval_results = []
model = EEGNet(num_channels=22, F_1=8, F_2=16, D=2)
criterion = nn.CrossEntropyLoss()                           
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for fold, (train_index, test_index) in enumerate(kf.split(data_tensor, labels_tensor)): # 이게 맞나
    train_subset = Subset(train_dataset, train_index)
    test_subset = Subset(train_dataset, test_index)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    if fold > 0:
        # 이전 폴드의 모델 상태 불러오기 (축적되는 학습을 하고 싶어서,,)
        model.load_state_dict(torch.load(f'model_fold_{fold-1}.pt'))

    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        print(f'Fold: {fold}, Epoch: {epoch}, Train Loss: {train_loss:.4f}')
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        # scheduler.step(test_loss) 학습률을 알아서 조정
        kf_results.append((test_loss, test_accuracy))
    
    plt.plot(kf_results) # 검증 데이터로 알아본 모델의 성능을 그래프로 나타내고자!
    
    # 현재 폴드의 모델 상태 저장 (다음에 불러 쓸 수 있게)
    torch.save(model.state_dict(), f'model_fold_{fold}.pt')
    # state_dict() : 파라미터와 값을 매핑하는 dictionary

#############################################################################
# Testing Model을 따로 쓸 때
model.load_state_dict(torch.load(f'model_fold_4.pt'))
eval_loss, eval_accuracy = evaluate_model(model, eval_loader, criterion)
eval_results.append((eval_loss, eval_accuracy))

avg_test_loss = np.mean([result[0] for result in kf_results])
avg_test_accuracy = np.mean([result[1] for result in kf_results])
avg_eval_loss = np.mean([result[0] for result in eval_results])
avg_eval_accuracy = np.mean([result[1] for result in eval_results])

print(f'Average KF Test Loss: {avg_test_loss:.4f}')
print(f'Average KF Test Accuracy: {avg_test_accuracy:.4f}')
print(f'Average Eval Loss: {avg_eval_loss:.4f}')
print(f'Average Eval Accuracy: {avg_eval_accuracy:.4f}')

plt.show()
