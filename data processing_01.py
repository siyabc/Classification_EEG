import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from mne.io import read_epochs_eeglab
from mne.time_frequency import psd_array_welch


def segment_eeg_data(data: np.ndarray, n: int = 3, overlap: int = 100):
    n_epochs, n_channels, n_times = data.shape
    segment_length = (n_times + overlap * (n - 1)) // n
    segments = []

    for i in range(n):
        start = i * (segment_length - overlap)
        end = min(start + segment_length, n_times)
        segment = data[:, :, start:end]
        segments.append(segment)

    if end < n_times:
        segment = data[:, :, -segment_length:]
        segments.append(segment)

    segmented_data = np.concatenate(segments, axis=0)
    return segmented_data


def process_all_set_files(input_folder: str, output_csv_folder: str, output_fig_folder: str, n: int = 3, overlap: int = 100):
    os.makedirs(output_csv_folder, exist_ok=True)
    # os.makedirs(output_fig_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".set"):
            set_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            print(f"\n📂 正在处理: {filename}")

            try:
                epochs = read_epochs_eeglab(set_path, verbose='ERROR')
                data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
                sfreq = epochs.info['sfreq']
                ch_names = epochs.ch_names

                segmented = segment_eeg_data(data, n=n, overlap=overlap)
                n_segments = segmented.shape[0]
                n_channels = segmented.shape[1]

                all_segment_dfs = []

                # plt.figure(figsize=(12, 6))

                for seg_idx in range(n_segments):
                    segment = segmented[seg_idx]  # shape: (n_channels, n_times)
                    rows = []

                    for ch_idx in range(n_channels):
                        ch_data = segment[ch_idx]
                        psd, freqs = psd_array_welch(ch_data, sfreq=sfreq, n_fft=len(ch_data), verbose='ERROR')

                        # 构建一个 DataFrame：每列一个通道，每行一个频率点
                        df = pd.DataFrame({
                            f"Segment{seg_idx+1}_Ch{ch_names[ch_idx]}_Freq(Hz)": freqs,
                            f"Segment{seg_idx+1}_Ch{ch_names[ch_idx]}_Power": psd
                        })
                        rows.append(df)

                        # if ch_idx < 2:
                            # plt.plot(freqs, psd, label=f"Seg{seg_idx+1} {ch_names[ch_idx]}")

                    # 横向拼接（避免 insert 多次）
                    segment_df = pd.concat(rows, axis=1)
                    all_segment_dfs.append(segment_df)

                # 所有段合并保存 CSV
                csv_df = pd.concat(all_segment_dfs, axis=1)
                csv_path = os.path.join(output_csv_folder, f"{base_name}_PSD.csv")
                csv_df.to_csv(csv_path, index=False)
                print(f"✅ PSD数据保存: {csv_path}")

                # 保存图像
                # plt.title(f"Power Spectrum - {base_name}")
                # plt.xlabel("Frequency (Hz)")
                # plt.ylabel("Power Spectral Density (V²/Hz)")
                # plt.grid(True, alpha=0.3)
                # plt.legend(loc='upper right', fontsize=8)
                # plt.tight_layout()
                # plot_path = os.path.join(output_fig_folder, f"{base_name}_PSD.png")
                # plt.savefig(plot_path, dpi=200)
                # plt.close()
                # print(f"✅ 功率谱图保存: {plot_path}")

            except Exception as e:
                print(f"❌ 处理失败: {e}")
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    input_folder = r'C:\Users\chenzhijia\Desktop\machine learning\sedation-restingstate\Sedation-RestingState'
    output_csv_folder = r'C:\Users\chenzhijia\Desktop\machine learning\sedation-restingstate\done'
    output_fig_folder = os.path.join(output_csv_folder, 'Power spectrum')

    process_all_set_files(input_folder, output_csv_folder, output_fig_folder)
