import numpy as np
import numpy.matlib
import struct

def get_file():
    '''获取 mnist 数据集的信息'''
    with open("C:/Users/12041/Desktop/01_PRbasis/src/train-images-idx3-ubyte", 'rb') as f:
        train = f.read()
        f.close()
    with open("C:/Users/12041/Desktop/01_PRbasis/src/train-labels-idx1-ubyte", 'rb') as f:
        label = f.read()
        f.close()
    idx3_data = struct.unpack_from('>4i', train)
    img_size = idx3_data[2] * idx3_data[3]  # 图片像素数
    idx1_data = struct.unpack_from('>2i', label)
    return train, label, idx3_data, idx1_data, img_size


def get_0_matices(n, *mat_shape):
    '''返回 n 个0矩阵，大小由 shape 列表/元组的值决定'''
    mat = []
    for i in range(n):
        mat.append(np.mat(np.zeros(shape=mat_shape[i])))
    return tuple(mat)


def get_image(start, end, train, img_size):
    '''获取包含 end - start + 1 张图片数据的 2 维数组'''

    # 创建空二维数组，存放 end - start + 1 张图片，每张图片有 img_size 个像素
    pic_data = np.empty((end - start + 1, img_size), dtype=int)

    # 读取格式：每次读取 img_size 个字节
    fmt_img = '>' + str(img_size) + 'B'

    # 读取起始位置：略过前 4 个总体信息和前 start - 1 张图片
    offset = 0+ struct.calcsize('>iiii') +(start - 1) * struct.calcsize(fmt_img)

    # 读取图片数据，存入数组
    for i in range(end - start + 1):
        img = np.array(struct.unpack_from(fmt_img, train, offset))
        pic_data[i] = img
        offset += struct.calcsize(fmt_img)

    # 返回数据
    return pic_data


def get_label(start, end, data):
    '''获取包含 end - start + 1 张图片标签信息的 1 维数组'''

    # 创建空 1 维数组，存放 end - start + 1 个标签信息zzlabel_data
    label_data = np.empty(shape=(end - start + 1), dtype=int)

    # 读取格式：一个字节
    fmt_label = '>B'

    # 读取起始位置：跳过前 2 个整数总体信息和前 start - 1 个标签信息
    offset = 0 + struct.calcsize('>2i') + (start - 1) * struct.calcsize(fmt_label)

    # 读取标签，放入数组
    for i in range(end - start + 1):
        label_data[i] = struct.unpack_from(fmt_label, data, offset)[0]
        offset += struct.calcsize(fmt_label)

    # 返回标签信息
    return label_data


def get_all_data(number, trainnum, testnum, img_size, label, train):
    '''获取训练特征向量 X (矩阵)、测试特征向量 X_test (矩阵)、测试标签向量(1 维)'''

    # 需要的数字
    m = number[0]
    n = number[1]

    # 训练图片、测试图片、测试标签
    X = np.empty(shape=(trainnum, img_size + 1), dtype=int)
    X_test = np.empty(shape=(testnum, img_size + 1), dtype=int)
    test_label = np.empty(testnum, dtype=int)

    # 按顺序读取图片，先放到训练集 X 里，再继续找一部分放到 test_data/label 里
    count = 0
    i = 1
    # 建立训练集，并规范化处理
    while count < trainnum:
        # get_label返回从第 i 个到第 i 个图片的标签数组，也就是返回一个只有一个元素-第 i 个图片标签的数组（列表）
        d = get_label(i, i, label)[0]
        if d == m:
            # 第一类样本：在 img 行向量的最后插入1-(x1,..xn,1)
            img = get_image(i, i, train, img_size)
            X[count] = np.insert(img, img_size, values=1, axis=1)
            count += 1
        elif d == n:
            # 第二类样本：在 img 行向量的最后插入1-(y1,..yn,1)-取反-(-y1,..-yn,-1)
            img = get_image(i, i, train, img_size)
            X[count] = (np.insert(img, img_size, values=1,axis=1)) * (-1)
            count += 1
        i += 1
    # 建立测试集
    count = 0
    while count < testnum:
        d = get_label(i, i, label)[0]
        if d == m or d == n:
            # 两类样本，在 img 行向量的最后插入1
            img = get_image(i, i, train, img_size)
            X_test[count] = np.insert(img, img_size, values=1, axis=1)
            test_label[count] = d
            count += 1
        i += 1
    # 转换为矩阵
    X = np.asmatrix(X)
    X_test = np.asmatrix(X_test)
    return X, X_test, test_label

