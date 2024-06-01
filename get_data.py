import gzip
import numpy as np


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 跳过前16个字节（文件头）
        f.read(16)
        # 读取剩余的图像数据
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(-1, 28, 28)
    return data


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # 跳过前8个字节（文件头）
        f.read(8)
        # 读取剩余的标签数据
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels