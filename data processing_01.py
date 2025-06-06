import os
import numpy as np
import pandas as pd
from mne.io import read_epochs_eeglab
from scipy.signal import stft
import sys
import random

# 定义频段范围（单位：Hz）
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

def gen_segments(data: np.ndarray, n: int = 3, overlap: int = 100):
    '''
    分段函数：将原始数据按时间进行分段处理
    :param data: 原始三维 EEG 数据，shape=(n_epochs, n_channels, n_times)
    :param n: 要分成几段
    :param overlap: 相邻段之间的重叠点数
    :return: 分段后的数据，shape=(n_epochs * n, n_channels, segment_length)
    '''
    n_epochs, n_channels, n_times = data.shape
    segment_length = (n_times + overlap * (n - 1)) // n
    segmented_data = []

    for i in range(n):
        start = i * (segment_length - overlap)
        end = min(start + segment_length, n_times)
        segment = data[:, :, start:end]
        segmented_data.append(segment)

    # 如果最后一段没有覆盖到数据末尾，补一段
    if end < n_times:
        segment = data[:, :, -segment_length:]
        segmented_data.append(segment)

    # 拼接所有分段，更新 shape -> (n_total_segments, n_channels, segment_length)
    segmented_data = np.concatenate(segmented_data, axis=0)
    return segmented_data

def extract_stft(segment: np.ndarray, fs: float = 500):
    '''
    使用 STFT 获取该段数据的频谱信息
    :param segment: 单个数据段，shape=(n_channels, segment_length)
    :param fs: 采样率
    :return: 频率数组 f，STFT 幅度，STFT 相位
    '''
    f, t, Zxx = stft(segment, fs=fs, nperseg=256, noverlap=128)
    amplitude = np.abs(Zxx)  # 幅度谱 shape: (n_channels, n_freqs, n_times)
    phase = np.angle(Zxx)    # 相位谱 shape: (n_channels, n_freqs, n_times)
    return f, amplitude, phase

def select_band_frequencies(frequencies, num_per_band=5):
    '''
    在五个频段中每段随机选择 num_per_band 个频率
    :param frequencies: 所有频率数组 f
    :param num_per_band: 每个频段要抽取的频率个数
    :return: 字典 {band: 频率列表}
    '''
    band_freqs = {}
    for band, (low, high) in FREQ_BANDS.items():
        band_range = frequencies[(frequencies >= low) & (frequencies <= high)]
        if len(band_range) >= num_per_band:
            selected = random.sample(list(band_range), num_per_band)
        else:
            selected = list(band_range)
        band_freqs[band] = selected
    return band_freqs

def process_folder(input_folder: str, output_folder: str, n: int = 3, overlap: int = 100, fs: float = 500):
    '''
    批量处理文件夹中的 .set 文件，提取频谱信息、幅度、相位，保存为 CSV
    :param input_folder: 输入 EEG .set 文件夹路径
    :param output_folder: 输出特征文件夹路径
    :param n: 分段数
    :param overlap: 重叠点数
    :param fs: 采样频率
    '''
    # 输出路径
    done_folder = os.path.join(output_folder, 'done')
    amplitude_folder = os.path.join(done_folder, 'amplitude')
    phase_folder = os.path.join(done_folder, 'phase')
    frequency_folder = os.path.join(done_folder, 'frequency')

    # 创建文件夹
    for folder in [amplitude_folder, phase_folder, frequency_folder]:
        os.makedirs(folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.set'):
            input_path = os.path.join(input_folder, file_name)
            print(f"Processing: {file_name}")

            # 读取 EEG 数据
            epochs = read_epochs_eeglab(input_path)
            data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

            # 分段处理
            segmented_data = gen_segments(data, n, overlap)

            # 初始化列表
            all_amplitude, all_phase = [], []
            all_freqs = None

            # 每个 segment 提取 STFT 特征
            for segment in segmented_data:
                f, amp, pha = extract_stft(segment, fs)
                if all_freqs is None:
                    all_freqs = f  # 保存频率数组
                # 保存每段的幅度和相位（flatten）
                all_amplitude.append(amp.reshape(amp.shape[0], -1))
                all_phase.append(pha.reshape(pha.shape[0], -1))

            # 拼接所有段的幅度和相位
            df_amplitude = pd.DataFrame(np.concatenate(all_amplitude, axis=1))
            df_phase = pd.DataFrame(np.concatenate(all_phase, axis=1))

            # 提取频段内的频率（不重复）
            selected_freqs = select_band_frequencies(all_freqs, num_per_band=5)
            # 展平频率字典为一个列表
            freq_list = []
            for band, freqs in selected_freqs.items():
                for f in freqs:
                    freq_list.append({'band': band, 'frequency': f})
            df_freqs = pd.DataFrame(freq_list)

            # 保存为 CSV
            base_name = file_name.replace('.set', '.csv')
            df_amplitude.to_csv(os.path.join(amplitude_folder, base_name), index=False)
            df_phase.to_csv(os.path.join(phase_folder, base_name), index=False)
            df_freqs.to_csv(os.path.join(frequency_folder, base_name), index=False)

            print(f"Saved: amplitude/phase/frequency for {file_name}")

# 主函数入口
if __name__ == '__main__':

    if sys.platform == "win32":
        # Windows 路径处理
        input_folder = r'C:\Users\chenzhijia\Desktop\machine learning\sedation-restingstate\Sedation-RestingState'
        output_folder = r'C:\Users\chenzhijia\Desktop\machine learning\sedation-restingstate\Output'
    else:
        # macOS/Linux 路径处理
        input_folder = r'../Sedation-RestingState'
        output_folder = r'../Output'

    n = 3
    overlap = 100

    # 执行批量处理
    process_folder(input_folder, output_folder, n, overlap)
