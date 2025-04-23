import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 设置随机种子和生成数据
np.random.seed(0)
X, y = make_moons(200, noise=0.20)

# 可视化数据
colors = ['CornflowerBlue', 'Tomato']
markers = ['o', '*']
plt.figure(figsize=(8, 6))
for i in range(2):
    plt.scatter(X[y == i, 0], X[y == i, 1],
                s=50,
                c=colors[i],
                marker=markers[i],
                label=f'Class {i}',
                edgecolors='None',
                alpha=0.8)
plt.title("Customized Scatter Plot of make_moons")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 定义神经网络类，使用Sigmoid激活函数
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
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


# 可视化决策边界
def plot_decision_boundary(pred_func):
    # 设置边界
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # 生成网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 预测每个点的类别
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制轮廓和训练样本
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='gray')
    plt.title("Decision Boundary with Sigmoid Activation")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


# 绘制决策边界
plot_decision_boundary(lambda x: nn.predict(x))