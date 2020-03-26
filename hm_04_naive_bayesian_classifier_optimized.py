import numpy as np
import struct
'''
朴素贝叶斯算法实现十分类
算法：P(wi|x)=P(x|wi)*P(wi)/P(x)
特点：图像二值化处理+朴素贝叶斯算法
难点：python中连续784个float相乘结果为inf
解决：概率整数化（不改变相对大小的放大），连乘时转化为python int计算
注意：ndarray与list的使用
        ndarray:数据类型为C类型，表示范围小；形状规则；切片灵活
        list:数据类型为python类型，表示范围大；形状任意；切片不灵活
优化：删去每张图片784个像素中的前60个和后64个，减小运算量（修改self.feature_len）
'''

class NB_Mnist_Classifier():
    '''朴素贝叶斯算法实现十分类（mnist数据集）'''

    def __init__(self):
        """
        self.feature_len：每张图片长度——优化接口
        self.prio_prob：先验概率——一维数组  P(wi) = list[i]
        self.cond_prob：类条件概率——三维数组  P(xk=0|wi) = list[i,k,0]

        self.train_num：训练集大小
        self.test_num：测试集大小
        self.tr_data：训练数据——二维数组
        self.tr_label：训练标签——一维数组
        self.tr_class_num：训练数据中各数字图片数目
        self.te_data：测试数据——二维数组
        self.te_label：训练标签——一维数组
        self.te_class_num：测试数据中各数字图片数目

        self.pred_corr_num：测试正确数目
        self.pred_result：测试结果（标签值）
        self.pred_prob：测试结果（概率值）
        """

        self.feature_len = 660
        self.prio_prob = None
        self.cond_prob = None

        self.train_num = None
        self.test_num = None
        self.tr_data = None
        self.te_data = None
        self.tr_label = None
        self.te_label = None
        self.tr_class_num = None
        self.te_class_num = None

        self.pred_corr_num = 0
        self.pred_result = None
        self.pred_prob = []


    @staticmethod
    def __binary_value(img):
        '''返回一维数组 img 的二值化结果'''
        for i in range(len(img)):
            if img[i] != 0:
                img[i] = 1
        return img

    @staticmethod
    def __get_one_il(fmt_img, fmt_lab, train, label, offset_img, offset_lab):
        '''返回读取的一组像素数组 img 、标签 lab'''
        img = np.array(struct.unpack_from(fmt_img, train, offset_img))
        lab = struct.unpack_from(fmt_lab, label, offset_lab)[0]
        img = NB_Mnist_Classifier.__binary_value(img)
        return img[60:720], lab

    def __calc_p_xinw(self, img, lab):
        '''返回 x 在 w 类的概率'''
        # 注意：prob 无默认类型，对两个乘数均 int()转化以指明 prob 为 int
        prob = int(self.prio_prob[lab])
        for i in range(self.feature_len):
            prob *= int(self.cond_prob[lab,i,img[i]])
        return prob

    def load(self, train_num, test_num):
        '''填入 tr_data、tr_label、te_data、te_label、train_num、test_num (int)'''

        # 0. 创建数组
        self.train_num = train_num
        self.test_num = test_num
        self.tr_data = np.zeros(shape=(self.train_num, self.feature_len), dtype=int)
        self.tr_label = np.zeros(shape=self.train_num, dtype=int)
        self.tr_class_num = np.zeros(shape=10, dtype=int)
        self.te_data = np.zeros(shape=(self.test_num, self.feature_len), dtype=int)
        self.te_class_num = np.zeros(shape=10, dtype=int)
        self.te_label = np.zeros(shape=self.test_num, dtype=int)

        # 1. 打开文件，获取二进制信息
        with open("C:/Users/12041/Desktop/01_PRbasis/src/train-images-idx3-ubyte", 'rb') as f:
            train = f.read()
        f.close()
        with open("C:/Users/12041/Desktop/01_PRbasis/src/train-labels-idx1-ubyte", 'rb') as f:
            label = f.read()
        f.close()
        idx3_data = struct.unpack_from('>4i', train)
        img_size = idx3_data[2] * idx3_data[3]  # 图片像素数
        idx1_data = struct.unpack_from('>2i', label)


        # 2. 提取 train_num 个训练样本
        # 通过 offset 和 fmt 读取 idx3/idx1 文件
        fmt_img = '>' + str(img_size) + 'B'
        fmt_lab = '>B'
        offset_img = 0 + struct.calcsize('>4i')
        offset_lab = 0 + struct.calcsize('>2i')
        for i in range(self.train_num):
            self.tr_data[i], self.tr_label[i] = NB_Mnist_Classifier.__get_one_il(fmt_img, fmt_lab, train, label, offset_img, offset_lab)
            self.tr_class_num[self.tr_label[i]] += 1

            offset_img += struct.calcsize(fmt_img)
            offset_lab += struct.calcsize(fmt_lab)


        # 3. 提取 test_num 个测试样本（按label分类）
        for i in range(self.test_num):
            self.te_data[i], self.te_label[i] = NB_Mnist_Classifier.__get_one_il(fmt_img, fmt_lab, train, label, offset_img, offset_lab)
            self.te_class_num[self.te_label[i]] += 1

            offset_img += struct.calcsize(fmt_img)
            offset_lab += struct.calcsize(fmt_lab)


        # 4. 结果报告
        print('0. 使用朴素贝叶斯分类器实现 0 - 9 手写数字识别\n'
              'Mnist测试集中共有 %d 张图片，%d 个标签，每张图片宽 %d 个像素，高 %d 个像素\n'
              % (idx3_data[1], idx3_data[2], idx3_data[3], idx1_data[1]))
        print('1. 成功读取 %d 组训练数据、 %d 组测试数据\n'
              '训练数据：\n数字0 :%4d 张，数字1 :%4d 张，数字2 :%4d 张，'
              '数字3 :%4d 张，数字4 :%4d 张\n数字5 :%4d 张，数字6 :%4d 张，'
              '数字7 :%4d 张，数字8 :%4d 张，数字9 :%4d 张'
              % (self.train_num, self.test_num, self.tr_class_num[0],
                 self.tr_class_num[1], self.tr_class_num[2], self.tr_class_num[3],
                 self.tr_class_num[4], self.tr_class_num[5], self.tr_class_num[6],
                 self.tr_class_num[7], self.tr_class_num[8], self.tr_class_num[9]))
        print('测试数据：\n数字0 :%4d 张，数字1 :%4d 张，数字2 :%4d 张，'
              '数字3 :%4d 张，数字4 :%4d 张\n数字5 :%4d 张，数字6 :%4d 张，'
              '数字7 :%4d 张，数字8 :%4d 张，数字9 :%4d 张'
              % (self.te_class_num[0], self.te_class_num[1], self.te_class_num[2],
                 self.te_class_num[3], self.te_class_num[4], self.te_class_num[5],
                 self.te_class_num[6], self.te_class_num[7], self.te_class_num[8],
                 self.te_class_num[9]))

    def fit(self):
        '''求先验概率 prio_prob、类条件概率 cond_prob（以 float 表示）'''

        # 0. 创建先验概率、类条件概率数组
        self.prio_prob = np.zeros(shape=10)
        self.cond_prob = np.zeros(shape=(10, self.feature_len, 2))

        # 1. 数字0出现次数；各图片index=k时0、1出现次数
        for i in range(self.train_num):
            img, lab = self.tr_data[i], self.tr_label[i]
            # P(wi) = n (/N)
            self.prio_prob[lab] += 1
            # n(xk=0) n(xk=1)
            for j in range(self.feature_len):
                self.cond_prob[lab, j, img[j]] += 1

        # 2. 计算
        for i in range(10):
            for j in range(self.feature_len):
                # P(xk=0|wi) = n(xk=0)/N(xk) * 100000 + 1
                n0, n1 = self.cond_prob[i, j, 0], self.cond_prob[i, j, 1]
                self.cond_prob[i, j, 0] = (float(n0)/float(n0+n1)) * 10000 + 1
                self.cond_prob[i, j, 1] = (float(n1)/float(n0+n1)) * 10000 + 1

        # 3. 结果报告
        print('\n2. 成功得到先验概率 P(wi) 、类条件概率 P(xk|wi)')
        print('先验概率 P(wi)：', self.prio_prob/self.train_num)
        print('类条件概率 P(xk=0、xk=1|wi)（取部分）：\n', self.cond_prob[0,90:100]/10002)

    def predict(self):
        '''计算后验概率 P(wi|x)，分类统计'''
        self.pred_corr_num = 0
        self.pred_result = np.zeros(shape=(self.test_num))
        self.pred_prob = []
        # 0. 对每一个测试元素 data 求属于 10 类数字的后验概率
        for i in range(self.test_num):
            img, lab = self.te_data[i], self.te_label[i]
            max_lab = 0
            max_prob = NB_Mnist_Classifier.__calc_p_xinw(self, img, 0)
            for j in range(1,10):
                prob = NB_Mnist_Classifier.__calc_p_xinw(self, img, j)
                if prob > max_prob:
                    max_prob = prob
                    max_lab = j
            self.pred_result[i] = max_lab
            self.pred_prob.append(max_prob)
            if max_lab == lab:
                self.pred_corr_num += 1

        # 4. 结果报告
        print('\n3. 成功测试 %4d 组数据，正确 %4d 组，总正确率为 %.3f'
              % (self.test_num, self.pred_corr_num, self.pred_corr_num / self.test_num))
        print('正确结果：', end='')
        for i in range(self.test_num):
            print('%3d'% self.te_label[i], end=' ')
        print('\n测试结果：', end='')
        for i in range(self.test_num):
            print('%3d' % self.pred_result[i], end=' ')


if __name__ == '__main__':
    bayes = NB_Mnist_Classifier()
    bayes.load(2000, 1000)
    bayes.fit()
    bayes.predict()
