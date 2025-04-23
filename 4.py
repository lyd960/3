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
