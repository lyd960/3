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