import os
import numpy as np
from mne.io import read_epochs_eeglab


def gen_corr_m(data: np.ndarray, n: int = 3, overlap: int = 100):
    '''
    :param data: 输入的三维数据矩阵，形状为 (40, 91, 2500)
    :param n: 将2500分成n份
    :param overlap: 每两份之间的重叠时间点数
    :return: 返回计算得到的相关系数矩阵，形状为 (40 * n, 91, 91)
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

    # 计算相关系数矩阵
    corr_matrices = np.zeros((segmented_data.shape[0], n_epochs, n_epochs))

    for i, segment in enumerate(segmented_data):
        corr_matrix = np.corrcoef(segment)
        corr_matrices[i] = corr_matrix

    return corr_matrices


def process_folder(input_folder: str, output_folder: str, n: int = 3, overlap: int = 100):
    '''
    批量处理一个文件夹中的 .set 文件，生成对应的相关系数矩阵并保存为 .npy 文件
    :param input_folder: 包含 .set 文件的输入文件夹路径
    :param output_folder: 保存 .npy 文件的输出文件夹路径
    :param n: 分段数量
    :param overlap: 每两段之间的重叠时间点数
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.set'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace('.set', '_corr_matrices.npy'))

            print(f"Processing file: {file_name}")

            # 读取 .set 文件
            epochs = read_epochs_eeglab(input_path)
            data = epochs.get_data()

            # 生成相关系数矩阵
            corr_m = gen_corr_m(data, n, overlap)

            # 保存为 .npy 文件
            np.save(output_path, corr_m)
            print(f"Saved correlation matrices to {output_path}")


if __name__ == '__main__':
    # 输入文件夹路径（包含 .set 文件）
    input_folder = r'C:\Users\chenzhijia\Desktop\机器学习\sedation-restingstate\Sedation-RestingState'

    # 输出文件夹路径（保存 .npy 文件）
    output_folder = r'C:\Users\chenzhijia\Desktop\机器学习\sedation-restingstate\Output'

    # 设置分段参数
    n = 3
    overlap = 100

    # 批量处理文件夹
    process_folder(input_folder, output_folder, n, overlap)
