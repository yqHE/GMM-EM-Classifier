#!/usr/bin/env python
# -*- encoding:utf-8 -*-


from pylab import *
import matplotlib.pyplot as plt


def get_data(root):
    """
    收集文件中的点, 生成X, Y以及标签T
    :param root:
    :return: 点集
    """
    X, Y, T = [], [], []
    with open(root, 'r', encoding='utf8') as f_w:
        line = f_w.readline()
        while line:
            line = line.strip()
            if not line:
                line = f_w.readline()
                continue
            if root.endswith('.txt'):
                x, y, tag = line.split()
                T.append(int(tag))
            elif root.endswith('.csv'):
                id, x, y = line.split(',')
            X.append(float(x))
            Y.append(float(y))
            line = f_w.readline()
    return [X, Y, T]


def generate_graph(list):
    """
    根据点集生成两种类别的曲线图
    :param list1: type1
    :param list2: type2
    :return:
    """
    # print(list)
    figure(figsize=(8, 6), dpi=80)

    # 设置坐标轴为原点
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    X1, X2, Y1, Y2 = [], [], [], []

    # 分类别
    for i in range(len(list[0])):
        if list[-1][i] == 1:
            X1.append(list[0][i])
            Y1.append(list[1][i])

        else:
            X2.append(list[0][i])
            Y2.append(list[1][i])
        i += 1

    # 绘制类别1曲线
    plt.scatter(X1, Y1, s=10, label='type1')

    # 绘制类别2曲线
    plt.scatter(X2, Y2, s=10, color='red', label='type2')

    plt.legend(loc='upper left')

    # 设置横轴的上下限
    plt.xlim(-2.0, 4.0)

    plt.xlabel('X')

    # 设置纵轴的上下限
    plt.ylim(-2.0, 4.0)

    plt.ylabel('Y')

    plt.show()


if __name__ == '__main__':
    train_root = './data/train.txt'
    test_root = './data/test.txt'
    dev_root = './data/dev.txt'

    data_list = get_data(train_root)
    # print(data_list[:5])

    generate_graph(data_list)


