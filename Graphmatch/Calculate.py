# from Graphmatch.utilsformatch import read_txt
import numpy as np
import networkx as nx
import math
# from Graphmatch.utilsformatch import read_txt
import numpy as np
from Graphmatch.Entity import Edge
from Graphmatch.utilsformatch import *
from Graphmatch.Match import *

from concurrent.futures import ThreadPoolExecutor


def calculate_grad(pts1,pts2):
    return (pts1[1]-pts2[1])/(pts1[0]-pts2[0])

def calculate_dist(pts1,pts2):
    return np.linalg.norm(pts1-pts2)

def calculate_edge_feat(pts1,pts2):
    return (calculate_dist(pts1,pts2),calculate_grad(pts1,pts2))

def calculate_adjacency_entropy(graph):
    num = graph.number_of_nodes()
    adj_matrix = nx.adjacency_matrix(graph).todense()
    # sum of degree
    k = np.zeros(num)
    for i in range(num):
        for j in range(num):
            k[i] += adj_matrix[i,j]
    # adjacency_degree
    A = np.zeros_like(k)
    for i in range(num):
        for j in range(num):
            if adj_matrix[i,j] != 0:
                A[i] += k[j]
    # selection_probability
    P = np.zeros_like(adj_matrix).astype(np.float32)
    for i in range(num):
        for j in range(num):
            if adj_matrix[i,j] != 0:
                P[i,j] = float(k[i] / A[j])
    # adjacency_information_entropy
    E = np.zeros_like(k)
    for i in range(num):
        for j in range(num):
            if adj_matrix[i,j] != 0:
                E[i] += -P[i,j] * math.log2(P[i,j])
    return E

def calculate_edge_value(img,p1,p2,num = 32):

    x1,y1 = p1
    x2,y2 = p2
    item = []
    for j in range(3):
        line_value = []
        for lambda_value in np.linspace(0, 1, num):  
            x = x1 + lambda_value * (x2 - x1)  
            y = y1 + lambda_value * (y2 - y1)  
            line_value.append(img[int(x),int(y),j])
        item.append(line_value)
    return np.array(item)

def calculate_cosine(a, b, c):  
    # 使用余弦定理计算角C  
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)  
    C = math.acos(cos_C)  
    return math.degrees(C)

def calculate_angles(A, B, C):
    # 计算三角形的边长  
    c = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)  
    a = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)      
    b = math.sqrt((C[0]-A[0])**2 + (C[1]-A[1])**2)  
    # 计算每个角的大小  
    angle_A = calculate_cosine(b, c, a)  
    angle_B = calculate_cosine(a, c, b)  
    angle_C = calculate_cosine(a, b, c)  
  
    # 将角度从小到大排序并返回  
    angles = np.array([angle_A, angle_B, angle_C])  
    # sorted_angles = np.sort(angles)  
    return angles

def calculate_sequence(graph):
    sequence = dict()
    for i in range(graph['graph'].number_of_nodes()):
        edge_sets_tmp = get_adjacent_tri(graph['tri'],i)
        edge_sets = order_tri(graph['tri'],edge_sets_tmp,i)
        sequence[i] = edge_sets
    return sequence

import numpy as np

import numpy as np

import numpy as np

def calculate_cosine(a, b, c):  
    # 使用余弦定理计算角C  
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)  
    C = math.acos(cos_C)  
    return math.degrees(C)

def calculate_angles(A, B, C):
    # 计算三角形的边长  
    c = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)  
    a = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)      
    b = math.sqrt((C[0]-A[0])**2 + (C[1]-A[1])**2)  
    # 计算每个角的大小  
    angle_A = calculate_cosine(b, c, a)  
    angle_B = calculate_cosine(a, c, b)  
    angle_C = calculate_cosine(a, b, c)  
  
    # 将角度从小到大排序并返回  
    angles = np.array([angle_A, angle_B, angle_C])  
    # sorted_angles = np.sort(angles)  
    return angles

def calculate_diff_angle(left_tris,left_pts,right_tris,right_pts):
    iter = max(len(left_tris),len(right_tris))
    turn = min(len(left_tris),len(right_tris))
    left_top_angles = []
    right_top_angles = []
    # 计算顶角度数
    for left_tri in left_tris:
        x,y,z = left_tri
        left_angle = calculate_angles(left_pts[x],left_pts[y],left_pts[z])
        left_top_angles.append(left_angle[0])
    for right_tri in right_tris:
        x,y,z = right_tri
        right_angle = calculate_angles(right_pts[x],right_pts[y],right_pts[z])
        right_top_angles.append(right_angle[0])
    np.array(left_top_angles)
    np.array(right_top_angles)
    sum_top_angel = np.sum(left_top_angles) + np.sum(right_top_angles)
    lam = 2.1 
    W_s = 0
    for t in range(iter):
        w_x = 0
        tri_simi = 0
        for i in range(turn):
            left_tri = left_tris[(i+t) % turn]
            right_tri = right_tris[i]
            x,y,z = left_tri
            left_angle = calculate_angles(left_pts[x],left_pts[y],left_pts[z])
            x,y,z = right_tri
            right_angle = calculate_angles(right_pts[x],right_pts[y],right_pts[z])
            ij_diff_angle = np.sum(np.abs(left_angle - right_angle))
            w_x = 1 - math.log(1 + lam * ij_diff_angle/180)
            tri_simi += (left_angle[0] + right_angle[0]) * w_x / sum_top_angel
        if W_s < tri_simi:
            W_s = tri_simi
    return W_s

def calculate_tri_simi(left_graph,left_pts,right_graph,right_pts):
    import time
    # start_time = time.time()
   
    left_sequence = calculate_sequence(left_graph)
    right_sequence = calculate_sequence(right_graph)
    tri_topo_simi = np.zeros((len(left_sequence),len(right_sequence)))
    # end_time = time.time()
    # print('#', end_time-start_time)
    # start_time = time.time()

    if(left_sequence!=dict() and right_sequence!=dict()):    
        tri_topo_simi = np.zeros((len(left_sequence),len(right_sequence)))
        for i in range(len(left_sequence)):
            for j in range(len(right_sequence)):
                left_tris = left_sequence[i]
                right_tris = right_sequence[j]
                wdiff = calculate_diff_angle(left_tris, left_pts, right_tris, right_pts)
                tri_topo_simi[i,j] = wdiff
    else:
        print(1)

    # end_time = time.time()
    # print('###', end_time-start_time)
    
    return tri_topo_simi

def calculate_support_simi(left_friends,right_friends):
    simi_matrix = np.zeros((len(left_friends),len(right_friends)))
    for i in range(len(left_friends)):
        for j in range(len(right_friends)):
            simi_matrix[i,j] = calculate_IoU(left_friends[i],right_friends[j])
    return simi_matrix

def calculate_IoU(left_list,right_list):
    n1 = len(set(left_list) & set(right_list))
    n2 = len(set(left_list) | set(right_list))
    if n2 == 0:
        return 0
    return n1/n2
    
def calculate_second_IoU(l_index,left_friends,left_graph,r_index,right_friends,right_graph,matches_set,alpha):
    f_order_iou = calculate_IoU(left_friends[l_index],right_friends[r_index])
    s_oder_iou = 0
    matched_nodes_left = []
    matched_nodes_right = []
    for node_pair in matches_set:
        if node_pair[0] in left_friends[l_index] and node_pair[1] in right_friends[r_index]:
            matched_nodes_left.append(node_pair[0])
            matched_nodes_right.append(node_pair[1])
    iou = 0
    n = 0
    for i,j in zip(matched_nodes_left,matched_nodes_right):
        l_tmp = left_graph['index'][i]
        r_tmp = right_graph['index'][j]
        iou += calculate_IoU(left_friends[l_tmp],right_friends[r_tmp])
        n += 1
    if n == 0:
        s_oder_iou = 0
    else:
        s_oder_iou = iou / n
    return alpha * f_order_iou + (1 - alpha) * s_oder_iou

def calculate_fundamental_matrix(left_pts,right_pts,matches):
    src_pts = []
    dst_pts = []
    for item in matches:
        src_pts.append(left_pts[item[0]][1:])
        dst_pts.append(right_pts[item[1]][1:])
    src_pts = np.array(src_pts).reshape(-1,2)
    dst_pts = np.array(dst_pts).reshape(-1,2)
    F, _ = cv2.findFundamentalMat(src_pts, dst_pts, method = cv2.FM_RANSAC,ransacReprojThreshold=0.9, confidence=0.99)
    return F