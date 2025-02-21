import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import pandas as pd
import glob


# 神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 64)  # 第二个全连接层
        self.fc3 = nn.Linear(64, output_dim)  # 输出层
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # 用于输出类别概率

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


# 加载数据和标签
def load_data_and_labels(npy_folder, mat_file):
    # 读取 datainfo.mat 文件，获取标签
    mat_data = loadmat(mat_file)
    labels = mat_data['datainfo'][:, 1].astype(int)  # 转换为整数类型，确保是 [1, 2, 3, 4] 这类标签

    # 获取所有样本的文件路径
    npy_files = glob.glob(npy_folder + '/*.npy')

    features = []

    # 遍历每个样本文件
    for npy_file in npy_files:
        data = np.load(npy_file)  # 读取当前样本的矩阵，形状为 [40*n, 91, 91]

        # 将每个样本的矩阵转换为向量（可以用 reshape 或 flatten）
        flattened_data = data.reshape(-1, 91 * 91)  # 展平为 (40*n, 91*91)

        # 使用 PCA 降维到 40 维
        pca = PCA(n_components=40)
        reduced_data = pca.fit_transform(flattened_data)

        # 取降维后的第一个样本作为特征（你可以根据需求选择哪一个）
        features.append(reduced_data[0])  # 假设只使用第一个样本的特征向量

    features = np.array(features)

    return features, labels


# 神经网络训练
def train_model(features, labels, input_dim, output_dim):
    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 将数据转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long) - 1  # 标签是 1 到 4，所以需要减 1
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long) - 1  # 标签是 1 到 4，所以需要减 1

    # 初始化模型、损失函数和优化器
    model = SimpleNN(input_dim=input_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_train_tensor)

        # 计算损失
        loss = criterion(outputs, y_train_tensor)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # 测试模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return model


# 保存特征和标签到 CSV
def save_to_csv(features, labels, output_file):
    # 将特征和标签组合成一个 DataFrame
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")


# 主函数
if __name__ == '__main__':
    # 设置路径
    npy_folder = r'C:\Users\chenzhijia\Desktop\machine learning\Output'  # .npy 文件存放的文件夹路径
    mat_file = r'C:\Users\chenzhijia\Desktop\machine learning\sedation-restingstate\Sedation-RestingState\datainfo.mat'  # .mat 文件的路径
    csv_file = r'C:\Users\chenzhijia\Desktop\machine learning\Final\features_labels.csv'  # 保存 CSV 文件的路径

    # 加载数据和标签
    features, labels = load_data_and_labels(npy_folder, mat_file)

    # 设置输入和输出维度
    input_dim = 40  # PCA 选择的特征维度，改为40
    output_dim = 4  # 标签类别数，1到4共4类

    # 训练模型
    model = train_model(features, labels, input_dim, output_dim)

    # 保存特征和标签到 CSV
    save_to_csv(features, labels, csv_file)
