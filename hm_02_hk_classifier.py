from hm_02_hk_func import *
import numpy as np
import numpy.matlib
import struct
'''
    H-K算法：同步更新松弛变量 b 和权向量 w 的算法，解决线性问题
    改变 train_num > iteration > p, number 权向量和正确率改变，幅度较小
    当样本数 l < n 特征空间维数，有多解（b 全大于 b0，每个b0的要求都满足）
    当样本数 l = n 特征空间维数，有唯一解（b 全大于 b0，每个b0的要求都唯一满足）
    当样本数 l > n 特征空间维数，有最优解（b 全大于 b0，但b0的要求一定被满足，由单调性）
    既然b0的要求总会被满足，不受样本数l硬性，何来唯一解-最优解的变化？
'''

def HK_sensor(number, train_num, test_num, p, iteration):
    '''
    基于集成 mnist 集的批量梯度 H-K 算法
    :param number: 元组，包含要分类的 2 个数字。如：(2, 3)
    :param train_num: 整数，希望训练用样本数。如：1000
    :param test_num: 整数，希望测试用样本数。如：100
    :param p: 小数（0 - 1），迭代步长。如：0.2
    :param iteration: 整数，最大迭代次数。如：200
    '''

    # part0.导入数据、初始矩阵设置

    # 读入图片、标签（二进制）
    train, label, idx3_data, idx1_data, img_size = get_file()

    # 初始化矩阵：b、w、e 、e（b0>0）
    w, b, e, e_pos = get_0_matices(4, (img_size + 1, 1), (train_num, 1), (img_size + 1, 1), (img_size + 1, 1))
    b += np.mat(np.ones(shape = (train_num, 1)))

    # 显示结果
    print("0：初始化设置")
    print("训练集图片总张数为 %d ，标签总数为 %d ，图片宽度和高度分别为：%d 和 %d\n"
          "本次测试的数字为 %d 和 %d \n"
          "训练的样本量为 %d，测试样本量为 %d \n"
          % (idx3_data[1], idx1_data[1], idx3_data[3], idx3_data[2], number[0], number[1], train_num, test_num))



    # part1. 初始化两向量：b0 w0

    # 读入特征向量（矩阵形式）
    X, X_test, test_label = get_all_data(number, train_num, test_num, img_size, label, train)

    # 计算 X 的伪逆：X_sharp = (XTX)-1XT
    X_sharp = np.linalg.pinv(X)

    # 计算 w 的初值: w0 = X_sharp * b0
    w = X_sharp * b

    # 显示结果
    print('1：b、w初始化')
    print('b0 矩阵：%4d x %4d  元素和：%12.3f  最大值：%7.3f  最小值：%7.3f'
          % (b.shape[0], b.shape[1], b.sum(), b.max(), b.min()))
    print('w0 矩阵：%4d x %4d  元素和：%12.3f  最大值：%7.3f  最小值：%7.3f'
          % (w.shape[0], w.shape[1], w.sum(), w.max(), w.min()))
    print('X  矩阵：%4d x %4d  元素和：%12.3f  最大值：%7.3f  最小值：%7.3f'
          % (X.shape[0], X.shape[1], X.sum(), X.max(), X.min()))
    print('X# 矩阵：%4d x %4d  元素和：%12.3f  最大值：%7.3f  最小值：%7.3f'
          % (X_sharp.shape[0], X_sharp.shape[1], X_sharp.sum(), X_sharp.max(), X_sharp.min()))



    # part234. 训练循环

    count = 0
    print('\n2：训练结果')
    while True:
        count += 1

        # part2.计算误差矩阵：e = Xw - b
        e = X * w - b
        e_pos = abs(e)

        # part3.判断是否结束：条件须尝试，正负条件不一定相同
        if e.max() < 0.01 and e.min() > -0.001:
            print('训练圆满成功')
            break
        elif e.max() < 0:
            print('系统线性不可分')
            break
        elif count >= iteration:
            print('在 %d 次迭代范围内，无法训练到指定精度' % iteration)
            break

        # part4.更新两向量：b = b + p * (e + |e|)  w = w + p * X_sharp|e|
        b += p * (e + e_pos)
        w += p * X_sharp * e_pos

    print('迭代次数：%d  步长：%.2f' % (count, p))
    print('e  矩阵：%4d x %4d  元素和：%12.3f  最大值：%7.3f  最小值：%7.3f'
          % (e.shape[0], e.shape[1], e.sum(), e.max(), e.min()))
    print('b  矩阵：%4d x %4d  元素和：%12.3f  最大值：%7.3f  最小值：%7.3f'
          % (b.shape[0], b.shape[1], b.sum(), b.max(), b.min()))
    print('w  矩阵：%4d x %4d  元素和：%12.3f  最大值：%7.3f  最小值：%7.3f'
          % (w.shape[0], w.shape[1], w.sum(), w.max(), w.min()))


    # part5. 测试

    correct = 0
    test_result = []
    test_value = []
    for i in range(test_num):
        Xt_w = X_test[i] * w
        test_value.append(Xt_w)
        if Xt_w  > 0 :
            test_result.append(number[0])
            if test_label[i] == number[0]:
                correct += 1
        elif Xt_w < 0:
            test_result.append(number[1])
            if test_label[i] == number[1]:
                correct += 1
    print('\n3：测试结果')
    print('本次测试样本数：%4d  正确数：%4d  正确率：%.3f' % (test_num, correct, correct/test_num))
    print('正确结果：', end='')
    for i in range(test_num):
        print('%5d'% test_label[i], end=' ')
    print('\n测试结果：', end='')
    for i in range(test_num):
        print('%5d' % test_result[i], end=' ')
    print('\n计算结果：', end='')
    for i in range(len(test_value)):
        print('%5.2f'% test_value[i], end=' ')


if __name__ == '__main__':
    HK_sensor((1,2), 785, 50, 0.3, 400)
