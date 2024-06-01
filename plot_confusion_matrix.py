from matplotlib import pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, labels_name, title='Confusion Matrix'):
    """
    绘制归一化的混淆矩阵的热图。
    参数：
    - cm: 2D数组，混淆矩阵。
    - labels_name: list，分类标签。
    - title: str，热图的标题。
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.figure(figsize=(8, 6))  # 设置图像尺寸
    plt.imshow(cm, interpolation='nearest', cmap='Blues')  # 在特定的窗口上显示图像，使用Blues色彩映射
    plt.title(title, fontsize=15)  # 图像标题
    plt.colorbar()

    num_local = np.arange(len(labels_name))
    plt.xticks(num_local, labels_name, rotation=45, fontsize=12)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=12)  # 将标签印在y轴坐标上

    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)

    # 在每个方格中添加数值注释
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > 0.5 else "black")

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()

