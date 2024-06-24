import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold
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

def import_EEG(file_name):
    f = open(file_name, 'r') # 받은 파일 읽어들이기
    lines = f.readlines() # 파일 내에 모든 줄을 읽어들여 리스트의 원소로 저장
    label_type = '0123456789' # 라벨이 1 digit number

    # 앞으로 생성하여 return 하고자 하는 array
    EEG_array = []
    label_array = []

    for l in lines: # 리스트 안의 요소를 하나씩 조사
        l = l.strip() # 줄 양 옆의 공백 제거
        if l in label_type: # 숫자라면 -> 라벨이라는 의미
            l = np.int64(l)
            label_array.append(l)
        else:
            data = json.loads(l) # l이 json 형태여서 (json 형태구나) 파이썬 객체로 변환
            data = np.float64(data)
            EEG_array.append(data)
    
    EEG_array = np.array(EEG_array) # epoch * timestamp * channel 형태의 3D array 생성
    label_array = np.array(label_array) # 0, 1, 2, 3 어떤 상상을 하는 지 나타내는 숫자들
    EEG_array = np.transpose(EEG_array, (0, 2, 1)) # epoch * channel * timestamp (EpochArray 형식 맞추려고)
    return EEG_array, label_array

def EEG_array_modifier(eeg_array, label_array):
    X = []
    y = []
    event_timepoints = []

    for (i, label) in enumerate(label_array):
        X.append(np.array(eeg_array[i])) # 2D ndarray 변환하여 삽입
        y.append(label) # 라벨 삽입
        event_timepoints.append(i) # 그냥 숫자 삽입. 결과적으로 0 1 2 3 4 ... 될 것
    events_array = np.array([[event_timepoints[i], 0, y[i]] for i in range(len(y))])
    return np.array(X), events_array

def EEG_to_epochs(eeg_array, label_array, sfreq = 500, event_id = {'Rest': 0, 'RightHand': 1, 'LeftHand': 2, 'Feet': 3}):
    # 우리가 꽂아서 사용한 채널(전극) 이름
    channels = ['F5', 'FC5', 'C5', 'CP5', 'P5', 'FC3', 'C3', 'CP3', 'P3', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'Cz', 'CPz', 'Pz', 'F2', 'FC2', 'C2', 'CP2', 'P2', 'FC4', 'C4', 'CP4', 'P4', 'F6', 'FC6', 'C6', 'CP6', 'P6']
    n_channels = len(channels)
    ch_types = ['eeg'] * n_channels
    # montage란, EEG 전극 이름과 두피 위의 센서의 상대적 위치를 나타낸다.
    # standard 1020은 정해진 하나의 규격 느낌인 듯 (?) 센서 위치 규격
    # 내장 EEG montage를 로드하는 것
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)
    info = info.set_montage(montage) # 설정한 montage 사용
    # 여기 피드 되는 두 개 array는 import_EEG에서 나온 친구들
    data, events = EEG_array_modifier(eeg_array, label_array)
    epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)
    return epochs

# Epoch Array에는 노치 필터 함수를 바로 적용 불가, 함수 구성
def apply_notch_filter(data, sfreq, freq=60.0, quality_factor=30.0):

    # 노치 필터의 주파수와 품질 인자 설정
    # 60Hz는 전원선 노이즈
    b, a = iirnotch(w0=freq, Q=quality_factor, fs=sfreq)
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

# print(os.getcwd())

EEG_array, label_array = import_EEG('[CYA]MI_four_1.txt') # 파일 읽어들이기
new_epoch = EEG_to_epochs(EEG_array, label_array) # 에폭 어레이 형성
# fig = new_epoch.plot(n_epochs=1, scalings = {"eeg": 500}, show=True, n_channels=32, event_color=dict({-1: "blue", 1: "red", 2: "yellow", 3: "green"})) # 기본적인 EEG 데이터 열람
# # 이벤트별로 온셋 타이밍 색깔로 보고 싶은데 코드가 적용이 안 되나 봄
# data, events = EEG_array_modifier(EEG_array, label_array)
# fig = mne.viz.plot_events(
#     events, event_id={'Rest': 0, 'RightHand': 1, 'LeftHand': 2, 'Feet': 3}, sfreq=new_epoch.info["sfreq"]
# )
# print("First few labels:", label_array[:10])
# print("First EEG epoch (first few samples):", EEG_array[0, :, :10])


##### 1. 필터링하기 #####
cutoff = 2
# highpass = new_epoch.copy().filter(l_freq=cutoff, h_freq=None)
highpass = new_epoch.filter(l_freq=cutoff, h_freq=None)
# with mne.viz.use_browser_backend("matplotlib"):
#     fig = highpass.plot(n_channels=32, n_epochs=1)
# fig.subplots_adjust(top=0.9)
# fig.suptitle(f"High-pass filtered at {cutoff} Hz", size="xx-large", weight="bold")

# ## 필터 양상 보기 ##
# filter_params = mne.filter.create_filter(
#     new_epoch.get_data(), new_epoch.info["sfreq"], l_freq=1, h_freq=None
# )
# mne.viz.plot_filter(filter_params, new_epoch.info["sfreq"], flim=(0.01, 5))
# plt.show()

## spectrum 그래프에 60Hz의 하모닉스에 표시해줌
# def add_arrows(axes):
#     for ax in axes:
#         freqs = ax.lines[-1].get_xdata()
#         psds = ax.lines[-1].get_ydata()
#         for freq in (60, 120, 180, 240):
#             idx = np.searchsorted(freqs, freq)
#             # get ymax of a small region around the freq. of interest
#             y = psds[(idx - 4) : (idx + 5)].max()
#             ax.arrow(
#                 x=freqs[idx],
#                 y=y + 18,
#                 dx=0,
#                 dy=-12,
#                 color="red",
#                 width=0.1,
#                 head_width=3,
#                 length_includes_head=True,
#             )

##### 2. 푸리에 변환 통해서, 내 에폭 안의 데이터들의 fq vs V spectrum 표시 #####
# compute_psd returns EpochSpectrum
# average : averages over channels
# amplitude : False thus Power spectrum (Amplitude spectrum if True)
# fig = new_epoch.compute_psd(fmax=250, method="welch", n_fft=4096).plot(
#     average=True, amplitude=False, exclude="bads"
# )

# epoch array data에 노치 필터를 적용하고 싶은데 지원하지 않는다고 한다.
# raw 에 적용한 거 보니까 60 헤르츠 하모닉스에서 파워가 확 감소하는 방향으로 수정됨

notch_data = new_epoch.get_data()
# 파라미터 차례대로 데이터 ndarray, 샘플링 주파수, 삭제할 주파수 영역, 품질 인자
notch_filtered = apply_notch_filter(notch_data, 500, 60, 30)
notch_filtered = apply_notch_filter(notch_filtered, 500, 120, 30)
notch_filtered = apply_notch_filter(notch_filtered, 500, 180, 30)
notch_filtered = apply_notch_filter(notch_filtered, 500, 240, 30)
new_epoch._data = notch_filtered
# fig = new_epoch.compute_psd(fmax=250).plot(
#     average=True, amplitude=False, exclude="bads"
# )
# fig.suptitle(f"Notch filtered", size="xx-large", weight="bold")

##### 3. 샘플링 레이트를 줄이고, ALIASING 방지하여 나타내기 #####
downsampled = new_epoch.copy().resample(sfreq=200) # sampling rate 500 -> 200

# n_ffts = [1024, int(round(1024 * 200 / new_epoch.info["sfreq"]))]
# fig, axes = plt.subplots(2, 1, sharey=True, layout="constrained", figsize=(10, 6))
# for ax, data, title, n_fft in zip(
#     axes, [new_epoch, downsampled], ["Original", "Downsampled"], n_ffts
# ):
#     fig = data.compute_psd(method="welch", n_fft=n_fft).plot(
#         average=True, amplitude=False, picks="data", exclude="bads", axes=ax
#     )
#     ax.set(title=title, xlim=(0, 300))

new_epoch = downsampled

##### 4. Artifact removal #####
# Checking artifacts
channels = ['F5', 'FC5', 'C5', 'CP5', 'P5', 'FC3', 'C3', 'CP3', 'P3', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'Cz', 
            'CPz', 'Pz', 'F2', 'FC2', 'C2', 'CP2', 'P2', 'FC4', 'C4', 'CP4', 'P4', 'F6', 'FC6', 'C6', 'CP6', 'P6']

artifact_picks = mne.pick_channels_regexp(channels, regexp='') # 그냥 채널들 다 고른다는 뜻이 되어 버림
# new_epoch.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# eog_evoked = create_eog_epochs(new_epoch).average()
# eog_evoked.apply_baseline(baseline=(None, -0.2))
# eog_evoked.plot_joint()

# High pass filtering and ICA
new_epoch = new_epoch.copy().filter(l_freq=1.0, h_freq=None)
ica = ICA(n_components=15, max_iter="auto", random_state=97) # ICA 객체 초기화 및 생성
ica.fit(new_epoch) # 필터링 된 데이터를 ICA 모델에 피팅시켜서 독립 성분들을 분리해낸다.
ica

# 추출된 독립 성분들이 원본 데이터의 '변동성'을 얼마나 잘 설명하는지
# 각 성분이 이 데이터에서 얼마나 중요한지, 중요도가 높다는 것은 artifact일 확률이 낮다는 것 (notion)
# 채널 타입이 EEG로 하나기 때문에, EEG에 대한 것만 출력되고, 기여도가 99%로 매우 높음
explained_var_ratio = ica.get_explained_variance_ratio(new_epoch)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
    )

# 분리된 독립 성분들 나란히 나타내주기
# ica.plot_sources(new_epoch, show_scrollbars=False)

# 이 플롯이 각각의 독립 성분의 특성을 다양하게 나타내주므로, 여기서 어떤 것이 노이즈인지를 파악한다. 
# ica.plot_properties(new_epoch, picks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# 어떤 독립 성분을 뺄지 정했으면, ica에 빼는 것으로 등록함
ica.exclude = [1]

ica.apply(new_epoch)

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

data = standardize_data(new_epoch.get_data())
data = normalize_data(data) # 둘 다 하니까 0%는 좀 심한데

channels = ['F5', 'FC5', 'C5', 'CP5', 'P5', 'FC3', 'C3', 'CP3', 'P3', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'Cz', 'CPz', 'Pz', 'F2', 'FC2', 'C2', 'CP2', 'P2', 'FC4', 'C4', 'CP4', 'P4', 'F6', 'FC6', 'C6', 'CP6', 'P6']
n_channels = len(channels)
ch_types = ['eeg'] * n_channels
montage = mne.channels.make_standard_montage('standard_1020')
info = mne.create_info(ch_names=channels, sfreq=500, ch_types=ch_types)
info = info.set_montage(montage) # 설정한 montage 사용
# 여기 피드 되는 두 개 array는 import_EEG에서 나온 친구들
data1, events = EEG_array_modifier(EEG_array, label_array)
event_id = {'Rest': 0, 'RightHand': 1, 'LeftHand': 2, 'Feet': 3}

normstan_epoch = mne.EpochsArray(data1, info, events, tmin=0, event_id=event_id)
# normstan_epoch.plot(n_epochs=1, scalings = {"eeg": 500}, show=True, n_channels=32, event_color=dict({-1: "blue", 1: "red", 2: "yellow", 3: "green"}))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

### 전처리한 Epoch Array 형태의 EEG 데이터를 tensor 형태로 변환
data_tensor = torch.tensor(data, dtype=torch.float32) # 얘는 이폭 어레이
labels_tensor = torch.tensor(label_array, dtype=torch.float32) # 얘는 넘파이 상태
# data_tensor = data_tensor.permute(0, 2, 1)
dataset = TensorDataset(data_tensor, labels_tensor)
train_size = int(0.6 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

class EEG_CNN(nn.Module):
    def __init__(self):
        super(EEG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=31, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5) # 과적합 방지
        self.linear1 = nn.Linear(76544, 64)
        self.linear2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    return avg_loss

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# K-Fold Cross-Validation
kfold_results = []
for train_index, test_index in kf.split(dataset):
    train_subset = Subset(dataset, train_index)
    test_subset = Subset(dataset, test_index)
    
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=8, shuffle=False)
    
    model = EEG_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        print(f'Epoch: {epoch} / Train Loss: {train_loss:.4f}')
    
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    kfold_results.append((test_loss, test_accuracy))

avg_test_loss = np.mean([result[0] for result in kfold_results])
avg_test_accuracy = np.mean([result[1] for result in kfold_results])

print(f'Average Test Loss: {avg_test_loss:.4f}')
print(f'Average Test Accuracy: {avg_test_accuracy:.4f}')

plt.show()