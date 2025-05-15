import os
import numpy as np
import pandas as pd
from mne.io import read_epochs_eeglab
from scipy.signal import stft
import sys

def gen_segments(data: np.ndarray, n: int = 3, overlap: int = 100):
    '''
    :param data: 输入的三维数据矩阵，形状为 (40, 91, 2500)
    :param n: 将2500分成n份
    :param overlap: 每两份之间的重叠时间点数
    :return: 返回分段后的数据，形状为 (40 * n, 91, segment_length)
    '''
    n_channels, n_epochs, n_times = data.shape

    # 每一段的长度（需要调整以确保覆盖所有点）
    segment_length = (n_times + overlap * (n - 1)) // n

    segmented_data = []

    for i in range(n):
        # 计算分段的起始和结束索引，确保完整覆盖数据
        start = i * (segment_length - overlap)
        end = min(start + segment_length, n_times)  # 防止超出数据长度

        segment_data = data[:, :, start:end]
        segmented_data.append(segment_data)

    # 如果最后一段没有覆盖到数据末尾，补上一段
    if end < n_times:
        segment_data = data[:, :, -segment_length:]  # 最后一段从末尾开始向前截取
        segmented_data.append(segment_data)

    # 将分段后的数据合并，形状为 (40 * 分段数量, 91, segment_length)
    segmented_data = np.concatenate(segmented_data, axis=0)

    return segmented_data


def extract_features_from_segment(segment: np.ndarray, fs: float = 500):
    '''
    从每个信号段中提取相位、频率和幅度特征
    :param segment: 输入信号段，形状为 (n_channels, segment_length)
    :param fs: 信号的采样频率，默认为500 Hz
    :return: 返回相位、频率和幅度的提取值
    '''
    n_channels, segment_length = segment.shape

    # 使用STFT进行频率、幅度和相位提取
    f, t, Zxx = stft(segment, fs=fs, nperseg=256, noverlap=128)

    # 提取相位、频率和幅度
    amplitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    freq_offset = np.abs(f - np.mean(f))
    frequency = np.tile(freq_offset, (n_channels, len(t)))

    # 转换为二维数据 (n_channels, n_frequencies * n_times)
    amplitude = amplitude.reshape(n_channels, -1)
    phase = phase.reshape(n_channels, -1)
    frequency = frequency.reshape(n_channels, -1)

    return amplitude, phase, frequency


def process_folder(input_folder: str, output_folder: str, n: int = 3, overlap: int = 100, fs: float = 500):
    '''
    批量处理一个文件夹中的 .set 文件，生成对应的特征并保存为 CSV 文件
    :param input_folder: 包含 .set 文件的输入文件夹路径
    :param output_folder: 保存特征数据的输出文件夹路径
    :param n: 分段数量
    :param overlap: 每两段之间的重叠时间点数
    :param fs: 信号采样频率
    '''
    # 创建用于保存特征的文件夹
    amplitude_folder = os.path.join(output_folder, 'amplitude')
    phase_folder = os.path.join(output_folder, 'phase')
    frequency_folder = os.path.join(output_folder, 'frequency')

    # 如果文件夹不存在，创建它们
    for folder in [amplitude_folder, phase_folder, frequency_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.set'):
            input_path = os.path.join(input_folder, file_name)

            print(f"Processing file: {file_name}")

            # 读取 .set 文件
            epochs = read_epochs_eeglab(input_path)
            data = epochs.get_data()

            # 生成信号段
            segmented_data = gen_segments(data, n, overlap)

            all_amplitude = []
            all_phase = []
            all_frequency = []

            # 从每个信号段提取特征
            for segment in segmented_data:
                amplitude, phase, frequency = extract_features_from_segment(segment, fs)

                # 将每段信号的幅度、相位和频率分别存入列表
                all_amplitude.append(amplitude)
                all_phase.append(phase)
                all_frequency.append(frequency)

            # 将所有的特征保存为 CSV 文件
            # 拼接并保存为 DataFrame
            df_amplitude = pd.DataFrame(np.concatenate(all_amplitude, axis=1))
            df_phase = pd.DataFrame(np.concatenate(all_phase, axis=1))
            df_frequency = pd.DataFrame(np.concatenate(all_frequency, axis=1))

            # 文件路径
            amplitude_file = os.path.join(amplitude_folder, file_name.replace('.set', '_amplitude.csv'))
            phase_file = os.path.join(phase_folder, file_name.replace('.set', '_phase.csv'))
            frequency_file = os.path.join(frequency_folder, file_name.replace('.set', '_frequency.csv'))

            # 保存为 CSV 文件
            df_amplitude.to_csv(amplitude_file, index=False)
            df_phase.to_csv(phase_file, index=False)
            df_frequency.to_csv(frequency_file, index=False)

            print(f"Saved features to {amplitude_file}, {phase_file}, {frequency_file}")


if __name__ == '__main__':

    if sys.platform == "win32":
        # Windows 路径处理
        input_folder = r'C:\Users\chenzhijia\Desktop\machine learning\sedation-restingstate\Sedation-RestingState'
        output_folder = r'C:\Users\chenzhijia\Desktop\machine learning\sedation-restingstate\Output'
    else:
        # macOS/Linux 路径处理
        input_folder = r'../Sedation-RestingState'
        output_folder = r'../Output'

    # 设置分段参数
    n = 3
    overlap = 100

    # 批量处理文件夹
    process_folder(input_folder, output_folder, n, overlap)
