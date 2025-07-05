import numpy as np
from Graphmatch.utilsformatch import *
from Graphmatch.Estimate import *
from Graphmatch.Calculate import *
import numba 
from numba import jit



def dual_threshing(left_points, left_adj_M, left_weight, left_graph, \
                    right_points, right_adj_M, right_weight, right_graph, \
                    index, labels, \
                    lowThreshRatio, highThreshRatio):
    H = RHO_homography(left_points, right_points, index)
    if H is None:
        # print("number of match:{}".format(len(index)))
        # print("H is None")
        # exit()
        return [] , [] , H
    error = calculate_error(H, left_points, right_points, index)
    
    lowThreshLeft = []
    highThreshRight = []
    for pair in index:
        lowThresh = [lowThreshRatio * left_points[pair[0]][3], lowThreshRatio * right_points[pair[1]][3]]
        highThresh = [highThreshRatio * left_points[pair[0]][4], highThreshRatio * right_points[pair[1]][4]]

        lowThreshLeft.append(lowThresh)
        highThreshRight.append(highThresh)

  

    error = iter(error)
    good_match = []

    select_match = []
    bad_match = []
    for i, (d1, d2) in enumerate(zip(error, error)):
        # print("{}--{},err:{}".format(labels[i][0],labels[i][1],np.around(d1 + d2,2),))
        if d1 < lowThreshLeft[i][0] and d2 < lowThreshLeft[i][1]:
            good_match.append((labels[i][0], labels[i][1]))
        elif d1 > highThreshRight[i][0] or d2 > highThreshRight[i][1]:
            bad_match.append((labels[i][0], labels[i][1]))
        else:
            select_match.append((labels[i][0],labels[i][1]))
    for i in range(len(select_match)):
        flag = False
        for j in range(len(good_match)):
            if left_adj_M[left_graph['index'][select_match[i][0]]][left_graph['index'][good_match[j][0]]] != 0 and right_adj_M[right_graph['index'][select_match[i][1]]][right_graph['index'][good_match[j][1]]] != 0:
                good_match.append(select_match[i])
                flag = True
                break
        if not flag:
            bad_match.append(select_match[i])
        
    ##特判处理没有匹配到的点：
    left_labels = [x[0] for x in good_match] + [x[0] for x in bad_match]
    right_labels = [x[1] for x in good_match] + [x[1] for x in bad_match]
    for i in range(len(left_points)):
        if left_points[i][0] not in left_labels:
            bad_match.append((left_points[i][0],-1))
    for i in range(len(right_points)):
        if right_points[i][0] not in right_labels:
            bad_match.append((-1,right_points[i][0]))
    return good_match, bad_match, H


def find_max_similarity(matrix):
    max_value = np.max(matrix)
    location = np.unravel_index(np.argmax(matrix), matrix.shape)
    return location[0], location[1], max_value


def find_best_matches(simi_matrix,hypothesis):
    matches = []
    matrix = simi_matrix.copy()
    row = matrix.shape[0]
    col = matrix.shape[1]
    if hypothesis is not None:
        for item in hypothesis:
            matches.append(item)
            for i in range(row):
                matrix[i][item[1]] = -1
            for i in range(col):
                matrix[item[0]][i] = -1
    # row_ind,col_ind = linear_sum_assignment(simi_matrix)
    # for i, j in zip(row_ind,col_ind):
    #     matches.append((i,j))
    while len(matches) < min(row,col):
        max_row, max_col, max_similarity = find_max_similarity(matrix)
        matches.append((max_row, max_col))

        for i in range(row):
            matrix[i][max_col] = -1
        for i in range(col):
            matrix[max_row][i] = -1

    return matches

def find_friends(left_friends, left_adj_matrix, left_graph, \
                 right_friends,right_adj_matrix,right_graph):
    for i in range(len(left_adj_matrix)):
        for j in range(len(left_adj_matrix)):
            if left_adj_matrix[i][j] != 0:
                left_friends[i].append(left_graph['labels'][j])
    for i in range(len(right_adj_matrix)):
        for j in range(len(right_adj_matrix)):
            if right_adj_matrix[i][j] != 0:
                right_friends[i].append(right_graph['labels'][j])
    
    # 假设 left_adj_matrix, right_adj_matrix 是 NumPy 数组，left_graph 和 right_graph 是包含 'labels' 键的字典
    left_adj_matrix = np.array(left_adj_matrix)
    right_adj_matrix = np.array(right_adj_matrix)

    left_labels = left_graph['labels']
    right_labels = right_graph['labels']

    # 使用 NumPy 的 nonzero 函数找到所有非零元素的索引
    left_nonzero_indices = np.nonzero(left_adj_matrix)
    right_nonzero_indices = np.nonzero(right_adj_matrix)

    # 创建朋友列表
    left_friends = [[] for _ in range(len(left_adj_matrix))]
    right_friends = [[] for _ in range(len(right_adj_matrix))]

    # 使用这些索引来填充朋友列表
    for i, j in zip(*left_nonzero_indices):
        left_friends[i].append(left_labels[j])

    for i, j in zip(*right_nonzero_indices):
        right_friends[i].append(right_labels[j])
    
def get_adjacent_tri(tri,index):
    adj_list = []
    for item in tri.simplices:
        if index in item:
            adj_list.append(list(item))
    return adj_list

def order_tri(tri,adj,index):
    item_edge = []
    ordered_edge = []
    # 点顺序
    for edge in adj:
        while edge[0] != index:
            item = edge.pop(0)
            edge.append(item)
        item_edge.append(edge)
    if(item_edge != []):
        first = item_edge.pop(0)
        # 边顺序
        ordered_edge.append(first)
        # 公共点
        while len(item_edge) != 0:
            item = item_edge.pop(0)
            flag = False
            for i in range(len(ordered_edge)):
                first = ordered_edge[i]
                # 待插入在first后面
                if item[1] == first[2]:
                    ordered_edge.insert(i+1,item)
                    flag = True
                    break
                # 待插入在first前面
                elif item[2] == first[1]:
                    ordered_edge.insert(i,item)
                    flag = True
                    break
            if flag == False:
                item_edge.append(item)
    else:
        print(1)
    return ordered_edge