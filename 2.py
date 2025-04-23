import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap
import sklearn.linear_model

# 设置随机种子和生成数据
np.random.seed(0)
X, y = make_moons(200, noise=0.20)


# 定义神经网络类，使用Sigmoid激活函数
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
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, learning_rate):
        num_examples = X.shape[0]

        # 反向传播
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 更新权重
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


# 训练神经网络
input_size = 2
hidden_size = 3
output_size = 2
nn = NeuralNetwork(input_size, hidden_size, output_size)

# 训练参数
learning_rate = 0.01
num_passes = 10000
print_every = 1000

for i in range(num_passes):
    # 前向传播
    probs = nn.forward(X)

    # 反向传播
    nn.backward(X, y, learning_rate)

    # 打印损失
    if i % print_every == 0:
        loss = nn.calculate_loss(X, y)
        print(f"迭代次数 {i}: 损失 {loss}")


# 自定义决策边界绘制函数
def plot_decision_boundary(pred_func):
    # 设置边界范围和网格间隔
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 网格点的预测
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 自定义填充颜色
    cmap_background = ListedColormap(['#a0c4ff', '#ffc9c9'])  # 浅蓝+浅橙
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.6)

    # 自定义点颜色和形状
    colors = ['CornflowerBlue', 'Tomato']
    markers = ['o', '*']
    for i in range(2):
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    s=60,
                    c=colors[i],
                    marker=markers[i],
                    label=f'Class {i}',
                    edgecolors='None',
                    alpha=0.9)


# 绘制神经网络决策边界
plt.figure(figsize=(8, 6))
plot_decision_boundary(lambda x: nn.predict(x))
plt.title("Neural Network with Sigmoid Activation")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("nn_sigmoid_decision_boundary.png", dpi=300)
plt.show()

# 对比逻辑回归结果
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)
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
#  A7-306
#  name:wuwenxing
#  time:2025/4/22 10:04
