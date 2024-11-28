"""
计算set文件的相关系数矩阵
"""

import numpy as np


def gen_corr_m(data:np.ndarray, n:int=1, overlap:int=0):
    '''
    :param data: set
    :param n: 将一条时间序列数据截断分成n份
    :param overlap: 相邻两份截断数据重叠的时间点数
    :return: correlation matrics (3D)
    '''
    return corr_m



if __name__ == '__main__':
    # load data
    data = ...
    n = 10
    overlap = 10
    corr_m = gen_corr_m(data, n, overlap)

    #save corr_m
    ...

