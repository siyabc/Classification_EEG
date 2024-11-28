"""
计算set文件的相关系数矩阵
"""

import numpy as np


#
# [1,2,3,4,5,6,7,8,9]
#
# [1,2,3][4,5,6][7,8,9] n=3, overlap = 0
#
# [1,2,3,4,5][3,4,5,6,7][5,6,7,8,9]  n=3, overlap = 3

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
    overlap = 50
    corr_m = gen_corr_m(data, n, overlap)

    #save corr_m
    ...

