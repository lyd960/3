##神经网络程序的修改
#雷雨典 324080203104 24机械一班 https://github.com/lyd960/3/blob/main/README.md
##1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 设置随机种子确保结果可重复
np.random.seed(0)

# 生成200个半月形数据点，添加20%的噪声
X, y = make_moons(200, noise=0.20)

# 数据可视化
colors = ['CornflowerBlue', 'Tomato']  # 定义两类数据的颜色
markers = ['o', '*']                  # 定义两类数据的标记形状

plt.figure(figsize=(8, 6))
for i in range(2):  # 分别绘制两类数据
    plt.scatter(X[y==i, 0], X[y==i, 1],  # X坐标和Y坐标
                s=50,                    # 点的大小
                c=colors[i],            # 颜色
                marker=markers[i],       # 形状
                label=f'Class {i}',      # 图例标签
                edgecolors='None',      # 无边缘颜色
                alpha=0.8)              # 透明度

plt.title("Customized Scatter Plot of make_moons")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重 - Xavier初始化
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    def forward(self, X):
        # 第一层计算
        self.z1 = np.dot(X, self.W1) + self.b1  # 线性变换
        self.a1 = self.sigmoid(self.z1)         # 非线性激活
        
        # 输出层计算
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax
        return self.probs
    def backward(self, X, y, learning_rate):
        num_examples = X.shape[0]
        
        # 输出层误差
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        
        # 计算W2和b2的梯度
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
        # 隐藏层误差（考虑sigmoid导数）
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(self.z1)
        
        # 计算W1和b1的梯度
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # 参数更新
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    def calculate_loss(self, X, y):
        num_examples = X.shape[0]
        correct_logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs) / num_examples
        return data_loss
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
# 网络架构参数
input_size = 2    # 输入特征维度
hidden_size = 3   # 隐藏层神经元数量
output_size = 2   # 输出类别数

nn = NeuralNetwork(input_size, hidden_size, output_size)

# 训练参数
learning_rate = 0.01
num_passes = 10000  # 迭代次数
print_every = 1000  # 每隔1000次打印损失

for i in range(num_passes):
    # 前向传播
    probs = nn.forward(X)
    
    # 反向传播和参数更新
    nn.backward(X, y, learning_rate)
    
    # 定期打印损失
    if i % print_every == 0:
        loss = nn.calculate_loss(X, y)
        print(f"迭代次数 {i}: 损失 {loss}")
def plot_decision_boundary(pred_func):
    # 设置绘图边界
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01  # 网格步长
    
    # 生成网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                         np.arange(y_min, y_max, h))
    
    # 预测网格中每个点的类别
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和数据点
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='gray')
    plt.title("Decision Boundary with Sigmoid Activation")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

# 绘制决策边界
plot_decision_boundary(lambda x: nn.predict(x))


##2
# 导入必要的库
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
from sklearn.datasets import make_moons  # 生成半月形数据集
from matplotlib.colors import ListedColormap  # 自定义颜色映射
import sklearn.linear_model  # 用于对比的逻辑回归模型

# 设置随机种子确保结果可重复
np.random.seed(0)

# 生成模拟数据
# make_moons生成两个半月形分布的数据集，适合测试非线性分类器
# 参数说明：
# n_samples=200: 生成200个样本
# noise=0.20: 添加20%的高斯噪声增加分类难度
X, y = make_moons(200, noise=0.20)

"""
神经网络类定义
使用Sigmoid作为激活函数，包含完整的训练逻辑
"""
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化神经网络参数
        :param input_size: 输入层维度
        :param hidden_size: 隐藏层神经元数量
        :param output_size: 输出层维度
        """
        # 权重初始化采用Xavier初始化方法，有助于缓解梯度消失/爆炸问题
        # W1: 输入层到隐藏层的权重矩阵，尺寸(input_size, hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        # b1: 隐藏层偏置向量，尺寸(1, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        # W2: 隐藏层到输出层的权重矩阵
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        # b2: 输出层偏置向量
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """
        Sigmoid激活函数
        公式: σ(x) = 1 / (1 + e^-x)
        将输入压缩到(0,1)区间
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Sigmoid函数的导数
        公式: σ'(x) = σ(x) * (1 - σ(x))
        这个特性使得计算非常高效
        """
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """
        前向传播过程
        :param X: 输入数据，尺寸(num_samples, input_size)
        :return: 输出层的概率分布
        """
        # 第一层计算：线性变换 + 激活函数
        self.z1 = np.dot(X, self.W1) + self.b1  # 线性变换
        self.a1 = self.sigmoid(self.z1)         # 非线性激活
        
        # 输出层计算：线性变换 + Softmax
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Softmax计算，将输出转换为概率分布
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    
    def backward(self, X, y, learning_rate):
        """
        反向传播过程
        :param X: 输入数据
        :param y: 真实标签
        :param learning_rate: 学习率
        """
        num_examples = X.shape[0]  # 样本数量
        
        # 输出层误差计算（Softmax的梯度）
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1  # 只对正确类别的梯度减1
        
        # 计算输出层权重和偏置的梯度
        dW2 = np.dot(self.a1.T, delta3)  # 隐藏层激活值的转置乘误差
        db2 = np.sum(delta3, axis=0, keepdims=True)  # 误差求和
        
        # 隐藏层误差传播（考虑sigmoid导数）
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(self.z1)
        
        # 计算隐藏层权重和偏置的梯度
        dW1 = np.dot(X.T, delta2)  # 输入数据的转置乘误差
        db1 = np.sum(delta2, axis=0)  # 误差求和
        
        # 参数更新：梯度下降
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def calculate_loss(self, X, y):
        """
        计算交叉熵损失
        :param X: 输入数据
        :param y: 真实标签
        :return: 平均损失值
        """
        num_examples = X.shape[0]
        # 只取正确类别对应的概率的对数
        correct_logprobs = -np.log(self.probs[range(num_examples), y])
        # 计算平均损失
        data_loss = np.sum(correct_logprobs) / num_examples
        return data_loss
    
    def predict(self, X):
        """
        预测函数
        :param X: 输入数据
        :return: 预测类别（概率最大的类别）
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)  # 返回概率最大的类别索引

# 实例化神经网络
input_size = 2    # 输入特征维度（x1, x2）
hidden_size = 3   # 隐藏层神经元数量
output_size = 2   # 输出类别数
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 训练参数设置
learning_rate = 0.01  # 学习率控制参数更新步长
num_passes = 10000    # 训练迭代次数
print_every = 1000    # 每隔多少次打印损失

# 训练循环
for i in range(num_passes):
    # 前向传播
    probs = nn.forward(X)
    
    # 反向传播和参数更新
    nn.backward(X, y, learning_rate)
    
    # 定期打印损失监控训练过程
    if i % print_every == 0:
        loss = nn.calculate_loss(X, y)
        print(f"迭代次数 {i}: 损失 {loss}")

def plot_decision_boundary(pred_func):
    """
    绘制决策边界可视化函数
    :param pred_func: 预测函数
    """
    # 设置绘图范围（数据范围+0.5的边界）
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01  # 网格步长
    
    # 生成网格点坐标矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 对每个网格点进行预测并reshape为网格形状
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 自定义背景颜色（决策区域）
    cmap_background = ListedColormap(['#a0c4ff', '#ffc9c9'])  # 浅蓝和浅橙色
    
    # 绘制决策区域
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.6)

    # 绘制数据点（两类不同样式）
    colors = ['CornflowerBlue', 'Tomato']  # 两类颜色
    markers = ['o', '*']                  # 两类标记形状
    for i in range(2):
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    s=60,                  # 点大小
                    c=colors[i],           # 颜色
                    marker=markers[i],     # 形状
                    label=f'Class {i}',    # 图例标签
                    edgecolors='None',     # 无边缘颜色
                    alpha=0.9)            # 透明度

# 绘制神经网络决策边界
plt.figure(figsize=(8, 6))
plot_decision_boundary(lambda x: nn.predict(x))
plt.title("Neural Network with Sigmoid Activation")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.tight_layout()  # 自动调整子图参数
plt.savefig("nn_sigmoid_decision_boundary.png", dpi=300)  # 保存高清图
plt.show()

# 对比：逻辑回归模型（线性分类器）
clf = sklearn.linear_model.LogisticRegressionCV()  # 带交叉验证的逻辑回归
clf.fit(X, y)  # 训练

# 绘制逻辑回归决策边界
plt.figure(figsize=(8, 6))
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logistic_regression_decision_boundary.png", dpi=300)
plt.show()


##3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. 生成 moon 形状的数据集
np.random.seed(0)
X, y = make_moons(200, noise=0.20)  # 这里正确定义了 X 和 y

# 2. 定义全局参数
num_examples = len(X)  # 现在 X 已经被定义
nn_input_dim = 2  # 输入维度
nn_output_dim = 2  # 输出维度
epsilon = 0.01  # 学习率
reg_lambda = 0.01  # 正则化参数


# 3. 定义辅助函数
def calculate_loss(model, X, y):  # 添加 X 和 y 作为参数
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']

    # 前向传播
    z1 = X.dot(W1) + b1
    a1 = 1 / (1 + np.exp(-z1))  # 使用 sigmoid 替代 tanh
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 计算损失
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    data_loss += (reg_lambda / 2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return data_loss / num_examples


def predict(model, x):
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']

    z1 = x.dot(W1) + b1
    a1 = 1 / (1 + np.exp(-z1))  # sigmoid
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)


# 4. 修改后的训练函数（使用 sigmoid）
def build_model(nn_hdim, num_passes=30000, print_loss=False):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    for i in range(num_passes):
        # 前向传播
        z1 = X.dot(W1) + b1
        a1 = 1 / (1 + np.exp(-z1))  # sigmoid
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 反向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        # 关键修改：sigmoid 的导数
        delta2 = delta3.dot(W2.T) * (a1 * (1 - a1))  # 替换 tanh 的导数
        dW1 = X.T.dot(delta2)
        db1 = np.sum(delta2, axis=0)

        # 添加正则化
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # 更新参数
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print(f"迭代 {i} 次后的损失值：{calculate_loss(model, X, y):.6f}")

    return model


# 5. 可视化函数
def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='k')
    plt.title("Decision Boundary with Sigmoid Activation")


# 6. 训练并可视化
model = build_model(3, print_loss=True)  # 隐藏层3个神经元
plt.figure(figsize=(8, 6))
plot_decision_boundary(lambda x: predict(model, x))
plt.savefig("sigmoid_decision_boundary.png", dpi=300)
plt.show()


##4
# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. 数据准备部分
np.random.seed(0)  # 设置随机种子保证结果可复现
X, y = make_moons(200, noise=0.20)  # 生成200个带噪声的半月形数据点

# 2. 定义神经网络参数
num_examples = len(X)  # 训练样本数量（200）
nn_input_dim = 2  # 输入层维度（二维坐标）
nn_output_dim = 2  # 输出层维度（二分类问题）
epsilon = 0.01  # 学习率（控制梯度下降步长）
reg_lambda = 0.01  # L2正则化系数（防止过拟合）


# 3. 修改后的sigmoid激活函数实现
def sigmoid(x):
    """Sigmoid激活函数：将输入压缩到(0,1)区间"""
    return 1 / (1 + np.exp(-x))


# 4. 损失函数计算（含L2正则化）
def calculate_loss(model, X, y):
    """计算交叉熵损失 + L2正则化项"""
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # 前向传播（使用sigmoid）
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)  # 修改点1：使用sigmoid替代tanh
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 计算交叉熵损失
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    # 添加L2正则化项（权重衰减）
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_loss / num_examples  # 返回平均损失


# 5. 预测函数
def predict(model, x):
    """使用训练好的模型进行预测"""
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # 前向传播（使用sigmoid）
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)  # 修改点2：使用sigmoid替代tanh
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)  # 返回预测类别


# 6. 核心训练函数（已修改为sigmoid）
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    """训练神经网络模型"""
    # 参数初始化（Xavier初始化）
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # 训练循环
    for i in range(num_passes):
        # 前向传播
        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)  # 使用sigmoid激活
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 反向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1  # 输出层误差

        # 计算梯度
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        # 关键修改点3：sigmoid的导数计算（a1*(1-a1)）
        delta2 = delta3.dot(W2.T) * (a1 * (1 - a1))  # 替换tanh的导数
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 添加L2正则化梯度
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # 参数更新
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2

        # 定期打印损失
        if print_loss and i % 1000 == 0:
            print(f"Loss after iteration {i}: {calculate_loss({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, X, y)}")

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


# 7. 决策边界可视化函数
def plot_decision_boundary(pred_func):
    """绘制模型决策边界"""
    # 设置绘图范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01

    # 生成网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测每个网格点
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和数据点
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='k')
    plt.xlabel("X1")
    plt.ylabel("X2")


# 8. 可视化不同隐藏层大小的效果
plt.figure(figsize=(16, 28))  # 创建大画布（16x28英寸）

# 测试不同的隐藏层节点数量
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]  # 从简单到复杂的网络结构

for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i + 1)  # 创建子图（5行2列布局）
    plt.title(f"Hidden Layer size: {nn_hdim}")  # 设置子图标题

    # 训练模型并绘制决策边界
    model = build_model(nn_hdim)  # 训练指定结构的模型
    plot_decision_boundary(lambda x: predict(model, x))  # 绘制决策边界

    # 添加坐标轴标签（只在底部子图显示）
    if i >= len(hidden_layer_dimensions) - 2:
        plt.xlabel("X1")
        plt.ylabel("X2")

plt.tight_layout()  # 自动调整子图间距
plt.savefig("ai_net_img_04.png", dpi=300)  # 保存高清图像
plt.show()
