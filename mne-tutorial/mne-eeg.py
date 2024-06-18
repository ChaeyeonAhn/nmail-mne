import numpy as np
import mne
import json
import matplotlib.pyplot as plt
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
    
    EEG_array = np.array(EEG_array) * 1e-9 # epoch * timestamp * channel 형태의 3D array 생성
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

def EEG_to_epochs(eeg_array, label_array, sfreq = 500, event_id = {'Rest': 0, 'Right Hand': 1, 'Left Hand': 2, 'Feet': 3}):
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

# print(os.getcwd())

EEG_array, label_array = import_EEG('[CYA]MI_four_1.txt')
new_epoch = EEG_to_epochs(EEG_array, label_array)
# modified_EEG, event = EEG_array_modifier(EEG_array, label_array)

new_epoch['Rest', 'Right Hand', 'Left Hand'].plot(n_epochs=10, show=True)
plt.show()

print("EEG_array shape:", EEG_array.shape)
print("Label array shape:", label_array.shape)
print("First few labels:", label_array[:10])
print("First EEG epoch (first few samples):", EEG_array[0, :, :10])