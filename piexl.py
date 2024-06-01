from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from get_data import *
from plot_confusion_matrix import *


# 加载训练数据
X_train = load_mnist_images('data/fashion/train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('data/fashion/train-labels-idx1-ubyte.gz')
X_train = X_train.reshape(-1, 784)

# 加载测试数据
X_test = load_mnist_images('data/fashion/t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('data/fashion/t10k-labels-idx1-ubyte.gz')
X_test = X_test.reshape(-1, 784)

# 打印数据集信息
print(f"训练集: {X_train.shape} 样本数，标签: {y_train.shape}")
print(f"测试集: {X_test.shape} 样本数，标签: {y_test.shape}")

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建SVM模型
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svm = SVC(kernel=kernel)
    # 训练模型
    svm.fit(X_train, y_train)
    # 预测测试集
    y_pred = svm.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    label_names = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    title = f"piexl Confusion Matrix (Kernel: {kernel})"
    plot_confusion_matrix(cm, label_names, title)
    # 打印分类报告和准确率
    print(kernel)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
