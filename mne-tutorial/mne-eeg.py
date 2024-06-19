import numpy as np
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
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
    
    EEG_array = np.array(EEG_array) * 0.000000016 # epoch * timestamp * channel 형태의 3D array 생성
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

# print(os.getcwd())

EEG_array, label_array = import_EEG('[CYA]MI_four_1.txt') # 파일 읽어들이기
new_epoch = EEG_to_epochs(EEG_array, label_array) # 에폭 어레이 형성
# fig = new_epoch.plot(n_epochs=1, show=True, n_channels=32) # 기본적인 EEG 데이터 열람

# print("First few labels:", label_array[:10])
# print("First EEG epoch (first few samples):", EEG_array[0, :, :10])


##### 1. 필터링하기 #####
cutoff = 2
highpass = new_epoch.copy().filter(l_freq=cutoff, h_freq=None)
with mne.viz.use_browser_backend("matplotlib"):
    fig = highpass.plot(n_channels=32, n_epochs=1)
fig.subplots_adjust(top=0.9)
fig.suptitle(f"High-pass filtered at {cutoff} Hz", size="xx-large", weight="bold")

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
fig = new_epoch.compute_psd(fmax=250, method="welch", n_fft=4096).plot(
    average=True, amplitude=False, exclude="bads"
)

# epoch array data에 노치 필터를 적용하고 싶은데 지원하지 않는다고 한다.
# raw 에 적용한 거 보니까 60 헤르츠 하모닉스에서 파워가 확 감소하는 방향으로 수정됨
# freqs = (60, 120, 180, 240)
# notch = new_epoch.copy().notch_filter(freqs=freqs)
# fig = notch.compute_psd(fmax=250).plot(
#     average=True, amplitude=False, exclude="bads"
# )
# fig.suptitle(f"Notch filtered", size="xx-large", weight="bold")

##### 3. 샘플링 레이트를 줄이고, ALIASING 방지하여 나타내기 #####
downsampled = new_epoch.copy().resample(sfreq=200) # sampling rate 500 -> 200

n_ffts = [4096, int(round(4096 * 200 / new_epoch.info["sfreq"]))]
fig, axes = plt.subplots(2, 1, sharey=True, layout="constrained", figsize=(10, 6))
for ax, data, title, n_fft in zip(
    axes, [new_epoch, downsampled], ["Original", "Downsampled"], n_ffts
):
    fig = data.compute_psd(method="welch", n_fft=n_fft).plot(
        average=True, amplitude=False, picks="data", exclude="bads", axes=ax
    )
    ax.set(title=title, xlim=(0, 300))

##### 4. Artifact removal #####
# Checking artifacts
channels = ['F5', 'FC5', 'C5', 'CP5', 'P5', 'FC3', 'C3', 'CP3', 'P3', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'Cz', 
            'CPz', 'Pz', 'F2', 'FC2', 'C2', 'CP2', 'P2', 'FC4', 'C4', 'CP4', 'P4', 'F6', 'FC6', 'C6', 'CP6', 'P6']

artifact_picks = mne.pick_channels_regexp(channels, regexp='') # 그냥 채널들 다 고른다는 뜻이 되어 버림
# new_epoch.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# High pass filtering and ICA
filt_epoch = new_epoch.copy().filter(l_freq=1.0, h_freq=None)
ica = ICA(n_components=15, max_iter="auto", random_state=97) # ICA 객체 초기화 및 생성
ica.fit(filt_epoch) # 필터링 된 데이터를 ICA 모델에 피팅시켜서 독립 성분들을 분리해낸다.
ica

# 추출된 독립 성분들이 원본 데이터의 '변동성'을 얼마나 잘 설명하는지
# 각 성분이 이 데이터에서 얼마나 중요한지
# 중요도가 높다는 것은 artifact일 확률이 낮다는 것
# 채널 타입이 EEG로 하나기 때문에, EEG에 대한 것만 출력되고, 기여도가 99%로 매우 높음
explained_var_ratio = ica.get_explained_variance_ratio(filt_epoch)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
    )

# 분리된 독립 성분들 나란히 나타내주기
# ica.plot_sources(new_epoch, show_scrollbars=False)

# 이 플롯이 각각의 독립 성분의 특성을 다양하게 나타내주므로, 여기서 어떤 것이 노이즈인지를 파악한다. 
# ica.plot_properties(new_epoch, picks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

# 어떤 독립 성분을 뺄지 정했으면, ica에 빼는 것으로 등록함
ica.exclude = [0, 2, 3, 4, 5, 6, 7, 8, 9, 12]
reconst_epoch = new_epoch.copy()
ica.apply(reconst_epoch)

new_epoch.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
reconst_epoch.plot(
    order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
)
del reconst_epoch
plt.show()