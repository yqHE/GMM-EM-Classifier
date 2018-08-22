#!/usr/bin/env python
# -*- encoding:utf-8 -*-
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
from sklearn.cluster import KMeans
from pylab import *
import matplotlib.pyplot as plt


def data_loader(root, has_tag):
    """
    读取训练、开发集('.txt')以及测试集('.csv')
    :param root:
    :return:
    """
    X, Y, T = [], [], []
    if root.endswith('.txt'):
        with open(root, 'r', encoding='utf8') as f_w:
            line = f_w.readline()
            while line:
                line = line.strip()
                if not line:
                    line = f_w.readline()
                    continue
                x, y, tag = line.split()
                X.append(float(x))
                Y.append(float(y))
                if has_tag:
                    T.append(int(tag))
                line = f_w.readline()
    elif root.endswith('.csv'):
        csv_file = csv.reader(open(root, 'r'))
        header = next(csv_file)
        for item in csv_file:
            X.append(float(item[1]))
            Y.append(float(item[2]))
    return X, Y, T


def divide_data(X, Y, T):
    """
    divide data by tag
    :param X:
    :param Y:
    :param T:
    :return:
    """
    X1, Y1, X2, Y2 = [], [], [], []
    for i in tqdm(range(len(T))):
        if T[i] == 1:
            X1.append(X[i])
            Y1.append(Y[i])
        else:
            X2.append(X[i])
            Y2.append(Y[i])
    return X1, Y1, X2, Y2


def Gaussian(data, mean, cov):
    """
    Guassian function
    :param data:
    :param mean:
    :param cov:
    :return:
    """

    dim = np.shape(cov)[0]

    # |cov|
    covdet = np.linalg.det(cov)

    # cov^-1
    # print('covdet is: {0}'.format(covdet))
    covinv = np.linalg.inv(cov)

    m = data - mean  # [4800, 2]

    # exp(-1/2*(x-mean)*cov*(x-mean))
    z = -0.5 * np.dot(np.dot(m, covinv), m.T)   # [4800, 4800]

    # N(x, mean, cov) = 1 / 口
    N = 1.0/(np.power(np.power(2*np.pi, dim)*abs(covdet), 0.5))*np.exp(np.diagonal(z))  # [4800,]

    return N


def GMM(data, K):
    """
    :param data: [4800, 2]二维矩阵
    :param K: 用几个高斯去拟合
    :return:
    """
    # reserve parameters
    para = []

    N = data.shape[0]
    dim = data.shape[1]

    kmeans = KMeans(n_clusters=K).fit(data)
    means = kmeans.cluster_centers_
    # means = np.random.rand(K, dim)  # [4, 2]
    means = np.array([[1.5, 0],
                      [0.5, 0.5],
                      [-0.5, 0],
                      [3, 0],
                      [1, 1.5],
                      [1, -1],
                      [2.5, 1],
                      [2, 2]])

    # 根据means求convs
    convs = np.random.rand(K, dim, dim)
    instance = np.zeros((K, N))
    for k in range(K):
        instance[k] = np.sum((data-means[k]) * (data-means[k]), axis=1) # [4800, 1]
    min_label = instance.argmin(0)
    for k in range(K):
        Xk = data[min_label==k]
        convs[k] = np.cov(Xk.T)

    # 多个高斯概率
    pis = [1.0/K] * K
    res = np.zeros((K, N))  # [4, 2400]

    loglikelyhood = 0
    oldloglikelyhood = float('-inf')

    likelyhoods = np.zeros((K, N))  # [4, 2400]

    itertimes = 0
    while np.abs(loglikelyhood - oldloglikelyhood) > 0.01:

        oldloglikelyhood = loglikelyhood

        for i in range(K):

            # E 步
            res[i] = pis[i] * Gaussian(data, means[i], convs[i]) # [2400,]

        gammas = res/np.array([np.sum(res, axis=0)]) # [4, 2400]

        NK = np.sum(gammas, axis=1)  # [4,]

        # M 步
        for i in range(K):
            # print(gammas[i])
            means[i] = np.dot(gammas[i], data)/NK[i]

            xdiff = data - means[i] # [2400, 2]

            convs[i] = np.dot(np.dot(xdiff.T, np.diag(gammas[i])), xdiff) / NK[i] # [2, 2]

        # 更新α
        pis = np.sum(gammas, axis=1)/N  # [4,]

        for i in range(K):
            likelyhoods[i] = np.multiply(pis[i], Gaussian(data, means[i], convs[i]))

        loglikelyhood = np.sum(np.log(np.sum(likelyhoods, axis=0)))

        para.append(loglikelyhood)
        itertimes += 1
        print(loglikelyhood)
    print('\n')
    return pis, means, convs, para, itertimes


def predict(data, pis, mean, conv, K):
    """
    预测800个点属于每个高斯分布的概率
    :param data: input, [N, dim]
    :param pis: α, weight, [K, 1]
    :param mean: [k, dim]
    :param conv: [k, dim, dim]
    :param K: number of Gaussians
    :return: probabilties
    """
    N = data.shape[0]
    Gauss = np.zeros((K, N))
    for i in range(K):
        Gauss[i] = Gaussian(data, mean[i], conv[i]) # [800,]
    prob = np.dot(np.diag(pis), Gauss)  # [4, 4800]
    probabilty = np.sum(prob, axis=0) # [800,]
    # print(probabilty)
    return probabilty


def get_tag(prob1, prob2, f_w):
    """
    decide the plot belong to which Gusssian and get the tag
    :param prob1:
    :param prob2:
    :return:
    """
    prob = prob1 - prob2
    tag = [1 if prob[i]>0 else 2 for i in range(len(prob))]

    ans = pd.read_csv('data/sample.csv')
    ans.iloc[:, 1] = tag
    ans.to_csv(f_w, index=0)
    return tag


def cal_perf(predict_t, gold_t):
    """
    calculate the performance of GMM model
    :param predict: list,
    :param gold: list,
    :return:
    """
    correct = 0
    for i in range(len(predict_t)):
        if predict_t[i] == gold_t[i]:
            correct += 1
        else:
            continue
    print('GMM perform in dev data %.4f' % (correct/len(predict_t)))


def generate_para(list1, iter1, list2, iter2):
    """
    describe the loglikelyhood
    :param list1:
    :param list2:
    :return:
    """
    X1 = [i for i in range(iter1)]
    X2 = [i for i in range(iter2)]
    figure(figsize=(8, 6), dpi=80)
    plt.scatter(X1, list1, s=10, label='type1')
    plt.scatter(X2, list2, s=10, label='type2')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    # 用8个高斯去拟合数据
    # input_gmm1 = np.array([X1+X2, Y1+Y2])
    # GMM(input_gmm1.T, 8)

    X, Y, T = data_loader('./data/train.txt', has_tag=True)
    X1, Y1, X2, Y2 = divide_data(X, Y, T)

    input_gmm1 = np.array([X1, Y1])
    pis1, means1, convs1, para1, itertimes1 = GMM(input_gmm1.T, 4)

    input_gmm2 = np.array([X2, Y2])
    pis2, means2, convs2, para2, itertimes2 = GMM(input_gmm2.T, 4)

    generate_para(para1, itertimes1, para2, itertimes2)

    # predict dev data
    X, Y, T = data_loader('./data/dev.txt', has_tag=True)
    dev_input = np.array([X, Y])
    prob1 = predict(dev_input.T, pis1, means1, convs1, 4)
    prob2 = predict(dev_input.T, pis2, means2, convs2, 4)
    predict_tag = get_tag(prob1, prob2, './data/dev.csv')
    cal_perf(predict_tag, gold_t=T)


    # predict test data
    X, Y, T = data_loader('./data/test.csv', has_tag=False)
    test_input = np.array([X, Y])
    prob1 = predict(test_input.T, pis1, means1, convs1, 4)
    prob2 = predict(test_input.T, pis2, means2, convs2, 4)
    get_tag(prob1, prob2, 'data/result.csv')



