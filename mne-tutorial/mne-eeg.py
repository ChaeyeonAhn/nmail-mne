import numpy as np
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score
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

# Epoch Array에는 노치 필터 함수를 바로 적용 불가, 함수 구성
def apply_notch_filter(data, sfreq, freq=60.0, quality_factor=30.0):

    # 노치 필터의 주파수와 품질 인자 설정
    # 60Hz는 전원선 노이즈
    b, a = iirnotch(w0=freq, Q=quality_factor, fs=sfreq)
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

def epochs_to_relative_psd_topo(epochs, cut_list, file_name='topomap',
                                event_list=['Rest', 'Right Hand'], save_path='topoplot.png'):
    # epochs = epochs.filter(l_freq=l_freq,h_freq=h_freq,method='iir',verbose=False)
    # when 'average', projection=True applies projection with apply_proj, projection=False immediately applies projection to the data
    
    epochs = epochs.set_eeg_reference('average', projection=True,verbose=False)

    fig,ax=plt.subplots(len(event_list)-1, len(cut_list), figsize=(30,30))
    fig.suptitle("{}".format(file_name, fontsize=16))

    for i, cut_range in enumerate(cut_list):
        lowcut, highcut = cut_range

        # REST (this will be used to subtract other psd)
        epoch_0 = epochs[event_list[0]]
        psd_0, freq_0 = epoch_0.compute_psd(fmin=lowcut, fmax=highcut).get_data(return_freqs=True)
        print(psd_0.shape)
        psd_0 = 10 * np.log10(psd_0)
        psd_0 = np.mean(psd_0, axis=0) # mean over sample dimension
        psd_0 = np.mean(psd_0, axis=1) # mean over time dimension

        for j in range(1,len(event_list)):
            epoch_j = epochs[event_list[j]]
            psd_j, freq_j = epoch_j.compute_psd(fmin=lowcut, fmax=highcut).get_data(return_freqs=True)#mne.time_frequency.psd_welch(epoch_j, lowcut, highcut,verbose=False)

            psd_j = 10 * np.log10(psd_j)
            psd_j = np.mean(psd_j, axis=0)
            psd_j = np.mean(psd_j, axis=1)

            psd_j -= psd_0

            topo_0 = mne.viz.plot_topomap(psd_j, epoch_j.info, axes=ax[j - 1][i], cmap=plt.cm.jet,
                                          show=False, names=epoch_j.info['ch_names'], vlim=(min(psd_j), max(psd_j)))

            ax[j - 1][i].set_title(event_list[j] + str(cut_range), fontdict={'fontsize': 24, 'fontweight': 'medium'})

    if save_path is not None:
        fig.savefig(save_path)  # save figure, but not working for some reason

    plt.show()

# print(os.getcwd())

EEG_array, label_array = import_EEG('[CYA]MI_four_1.txt') # 파일 읽어들이기
new_epoch = EEG_to_epochs(EEG_array, label_array) # 에폭 어레이 형성
fig = new_epoch.plot(n_epochs=1, show=True, n_channels=32, event_color=dict({-1: "blue", 1: "red", 2: "yellow", 3: "green"})) # 기본적인 EEG 데이터 열람
# 이벤트별로 온셋 타이밍 색깔로 보고 싶은데 코드가 적용이 안 되나 봄
data, events = EEG_array_modifier(EEG_array, label_array)
fig = mne.viz.plot_events(
    events, event_id={'Rest': 0, 'RightHand': 1, 'LeftHand': 2, 'Feet': 3}, sfreq=new_epoch.info["sfreq"]
)
# print("First few labels:", label_array[:10])
# print("First EEG epoch (first few samples):", EEG_array[0, :, :10])


##### 1. 필터링하기 #####
cutoff = 2
# highpass = new_epoch.copy().filter(l_freq=cutoff, h_freq=None)
highpass = new_epoch.filter(l_freq=cutoff, h_freq=None)
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

notch_data = new_epoch.get_data()
# 파라미터 차례대로 데이터 ndarray, 샘플링 주파수, 삭제할 주파수 영역, 품질 인자
notch_filtered = apply_notch_filter(notch_data, 500, 60, 30)
notch_filtered = apply_notch_filter(notch_filtered, 500, 120, 30)
notch_filtered = apply_notch_filter(notch_filtered, 500, 180, 30)
notch_filtered = apply_notch_filter(notch_filtered, 500, 240, 30)
new_epoch._data = notch_filtered
fig = new_epoch.compute_psd(fmax=250).plot(
    average=True, amplitude=False, exclude="bads"
)
fig.suptitle(f"Notch filtered", size="xx-large", weight="bold")

##### 3. 샘플링 레이트를 줄이고, ALIASING 방지하여 나타내기 #####
downsampled = new_epoch.copy().resample(sfreq=200) # sampling rate 500 -> 200

n_ffts = [1024, int(round(1024 * 200 / new_epoch.info["sfreq"]))]
fig, axes = plt.subplots(2, 1, sharey=True, layout="constrained", figsize=(10, 6))
for ax, data, title, n_fft in zip(
    axes, [new_epoch, downsampled], ["Original", "Downsampled"], n_ffts
):
    fig = data.compute_psd(method="welch", n_fft=n_fft).plot(
        average=True, amplitude=False, picks="data", exclude="bads", axes=ax
    )
    ax.set(title=title, xlim=(0, 300))

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
# eog_indices, eog_scores = ica.find_bads_eog(new_epoch)
# ica.exclude = eog_indices

# ica.plot_scores(eog_scores)
# ica.plot_properties(new_epoch, picks=eog_indices)
# ica.plot_sources(new_epoch, show_scrollbars=False)
# ica.plot_sources(eog_evoked)

 # ica apply를 하면 신호가 아예 바뀌어서 copy
reconst_epoch = new_epoch.copy()
ica.apply(reconst_epoch)

epochs_to_relative_psd_topo(reconst_epoch, cut_list) # What is this cut_list?

# new_epoch.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
# reconst_epoch.plot(
#     order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False
# )

# def standardize_data(data):
#     n_epochs, n_channels, n_timepoints = data.shape
#     standardized_data = np.zeros((n_epochs, n_channels, n_timepoints))
#     scaler = StandardScaler()
#     for channel in range(n_channels): # 채널별로 다 표준화
#         scaled_data = scaler.fit_transform(data[:, channel, :].T).T
#         standardized_data[:, channel, :] = scaled_data
#     return standardized_data

# def normalize_data(data):
#     n_epochs, n_channels, n_timepoints = data.shape
#     normalized_data = np.zeros((n_epochs, n_channels, n_timepoints))
#     scaler = MinMaxScaler()
#     for channel in range(n_channels):
#         scaled_data = scaler.fit_transform(data[:, channel, :].T).T
#         normalized_data[:, channel, :] = scaled_data
#     return normalized_data

# data = standardize_data(reconst_epoch.get_data())
# data = normalize_data(data)

# reconst_epoch = EEG_to_epochs(data, label_array)

# del reconst_epoch

## 이왕 ICA로 전처리 해본 김에 이걸로 특징 추출 해보자!
## 이제부터는 CSP를 이용하여 decoding
## 아래는 훈련용 데이터를 만들고 (그냥 시간 기준으로 자름), numpy 배열 객체로 뽑기
reconst_epoch_train = reconst_epoch.copy().crop(tmin=1.0, tmax=2.0)
reconst_epoch_data = reconst_epoch.get_data(copy=False)
reconst_epoch_tdata = reconst_epoch_train.get_data(copy=False)

# # 오히려 ICA 안 한 애들로 학습한 것이 정확도가 더 높다 큰 차이는 아니지만...!
# new_epoch_train = new_epoch.copy().crop(tmin=1.0, tmax=2.0)
# new_epoch_data = new_epoch.get_data(copy=False)
# new_epoch_tdata = new_epoch_train.get_data(copy=False)

# 내가 데이터를 어떻게 쪼개서 교차 검증에 쓸 건지
# 10개로 쪼갤 거고, 테스트 셋은 20%, 나머지는 훈련용 데이터
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(reconst_epoch_tdata)
# cv_split = cv.split(new_epoch_tdata)

# LDA 구성
# LDA는 주어진 데이터의 각 클래스 간에 제일 큰 분리가 일어나도록 그 분류하는 '직선'을 찾는 알고리즘
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
scaler = StandardScaler()

# CSP와 LDA를 함께 하도록 도와주는 파이프라인, 참고로 SVM 써서 비선형
# 스케일러 써도 크게 안 달라짐
clf = Pipeline([
    ("CSP", csp), 
    ("Scaler", scaler),
    ("LDA", lda)])

# 교차 검증을 통해서 내가 만든 분류 모델의 점수를 배열에 저장하여 반환
# 여기서 쓰는 CSP는 모델의 성능을 평가하기 위함
# 여기는 데이터 fragment를 가지고 함
scores1 = cross_val_score(clf, reconst_epoch_tdata, label_array, cv=cv, n_jobs=None)
# scores2 = cross_val_score(clf, new_epoch_tdata, label_array, cv=cv, n_jobs=None)

# 레이블 어레이에서 0번과 각 요소들이 똑같은 경우의 수를 계산하여 평균을 냄
class_balance = np.mean(label_array == label_array[0])

# 다수의 빈도수를 설정
class_balance = max(class_balance, 1.0 - class_balance)

# 얼마나 잘 검증하는지 / 기준 비율 (다수의 비율)
print(f"****** Classification accuracy (w/ ICA): {np.mean(scores1) * 100}% ******")
# print(f"****** Classification accuracy (no ICA): {np.mean(scores2) * 100}% ******")


# 전체 데이터에 대해서 CSP 학습! 여기서는 '진짜' 학습
csp.fit_transform(reconst_epoch_data, label_array)
# csp.fit_transform(new_epoch_data, label_array)

csp.plot_patterns(reconst_epoch.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
# csp.plot_patterns(new_epoch.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

# # 시간에 따른 퍼포먼스 시각적 열람하기
# sfreq = new_epoch.info["sfreq"] #500

# # 슬라이딩 윈도우의 크기를 지정
# # 슬라이딩 윈도우란, 데이터 위에 놓고 조금씩 움직여가며 분석을 수행할 때 쓰는 도구
# # 순차적인 신호 변화나 국소적인 특징을 알아볼 때 유용하다.
# w_length = int(sfreq * 0.5)  # running classifier: window length
# w_step = int(sfreq * 0.1)  # running classifier: window step size
# w_start = np.arange(0, new_epoch_data.shape[2] - w_length, w_step)

# scores_windows = []

# for train_idx, test_idx in cv_split: # 자세히는 모르겠지만 훈련 / 테스트 데이터로 나뉘어 있음
#     y_train, y_test = label_array[train_idx], label_array[test_idx]

#     X_train = csp.fit_transform(new_epoch_tdata[train_idx], y_train) # 얘는 훈련 데이터에 학습 후 학습된 모델 바탕으로 변환
#     X_test = csp.transform(new_epoch_tdata[test_idx]) # 얘는 테스트 데이터에 앞전에 학습한 모델 가지고 변환만.

#     # fit classifier
#     lda.fit(X_train, y_train)

#     # running classifier: test classifier on sliding window
#     score_this_window = []
#     for n in w_start:
#         X_test = csp.transform(new_epoch_data[test_idx][:, :, n : (n + w_length)]) 
#         score_this_window.append(lda.score(X_test, y_test))
#     scores_windows.append(score_this_window)

# # Plot scores over time
# w_times = (w_start + w_length / 2.0) / sfreq + new_epoch.tmin

# plt.figure()
# plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
# plt.axvline(0, linestyle="--", color="k", label="Onset")
# plt.axhline(0.5, linestyle="-", color="k", label="Chance")
# plt.xlabel("time (s)")
# plt.ylabel("classification accuracy")
# plt.title("Classification score over time")
# plt.legend(loc="lower right")
# plt.show()
plt.show()