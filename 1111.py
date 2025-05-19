import matplotlib
matplotlib.use('TkAgg')

from scipy.signal import stft
import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_epochs_eeglab

# 读取 EEG 数据
epochs = read_epochs_eeglab(
    r'C:\Users\chenzhijia\Desktop\machine learning\sedation-restingstate\Sedation-RestingState\02-2010-anest- 20100210 16.003.set'
)
data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

# 选第1个epoch、第1个通道的数据
signal = data[0, 0, :]  # shape: (2500,)

# 计算 STFT
f, t, Zxx = stft(signal, fs=500, nperseg=256, noverlap=128)
amplitude = np.abs(Zxx)
phase = np.angle(Zxx)

# ⚠️ 为了图像清晰，只画部分频率（例如前10个）
num_freqs_to_plot = 10

# 幅度折线图
plt.figure(figsize=(10, 5))
for i in range(num_freqs_to_plot):
    plt.plot(t, amplitude[i, :], label=f'{f[i]:.1f} Hz')
plt.title('Amplitude over Time (Line Plot)')
plt.xlabel('Time [sec]')
plt.ylabel('Amplitude')
plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.tight_layout()
plt.show()

# 相位折线图
plt.figure(figsize=(10, 5))
for i in range(num_freqs_to_plot):
    plt.plot(t, phase[i, :], label=f'{f[i]:.1f} Hz')
plt.title('Phase over Time (Line Plot)')
plt.xlabel('Time [sec]')
plt.ylabel('Phase (radian)')
plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.tight_layout()
plt.show()
