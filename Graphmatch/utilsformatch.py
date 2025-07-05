import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
import numba 
from numba import jit
import random

def func(x):
    return -x**2+2*x

def exam_F(src_pts,dst_pts,F):
    homo_src = np.array([[x,y,1] for [x,y] in src_pts])
    homo_dst = np.array([[x,y,1] for [x,y] in dst_pts])
    M = np.matmul(np.matmul(homo_dst,F),homo_src.T)
    ## 8个点是指matcher可靠的个数，对角线元素即为误差
    print(np.diag(M))

def exam_Epiline(train_pts,query_pts,src_pts,dst_pts,index_pts,F):
    homo_src = []
    homo_dst = []
    for i in range(len(index_pts)):
        homo_src.append([src_pts[int(index_pts[i][0])][0],src_pts[int(index_pts[i][0])][1],1])
        homo_dst.append([dst_pts[int(index_pts[i][1])][0],dst_pts[int(index_pts[i][1])][1],1])
    lines1 = cv2.computeCorrespondEpilines(train_pts.reshape(-1,1,2),1,F)
    lines1 = lines1.reshape(-1,3)
    homo_src = np.array(homo_src)
    homo_dst = np.array(homo_dst)
    # lines1 = np.matmul(F,homo_src.T).T
    lines2 = cv2.computeCorrespondEpilines(query_pts.reshape(-1,1,2), 2,F)
    lines2 = lines2.reshape(-1,3)
    for i in range(len(index_pts)):
        d = abs(np.sum(np.multiply(homo_dst[i],lines1[i]))) / math.sqrt(np.sum(lines1[i]**2)) + abs(np.sum(np.multiply(homo_src[i],lines2[i]))) / math.sqrt(np.sum(lines2[i]**2))

def init_graph(points):
    # 生成三角形
    tri = Delaunay(points[:,1:3]) 
    pos = dict()
    index = dict()
    labels = dict()
    for i,point in enumerate(points):
        pos[i] = (point[1],768-point[2])
        labels[i] = point[0]
        index[point[0]] = i
    # 创建图
    graph = nx.Graph()
    for i in range(len(points)):
        graph.add_node(i,label = labels[i])
    G = dict()
    G['graph'] = graph
    G['pos'] = pos
    G['labels'] = labels
    G['index'] = index
    G['tri'] = tri
    ds = []

    # 三角网
    for edge in tri.simplices:
        x, y, z = edge
        d1 = math.sqrt(pow(points[x][0] - points[y][0],2)+pow(points[x][1] - points[y][1],2))
        ds.append(d1)
        d2 = math.sqrt(pow(points[y][0] - points[z][0],2)+pow(points[y][1] - points[z][1],2))
        ds.append(d2)
        d3 = math.sqrt(pow(points[x][0] - points[z][0],2)+pow(points[x][1] - points[z][1],2))
        ds.append(d3)
    ds = np.array(ds)
    minv = np.min(ds)
    maxv = np.max(ds)
    for edge in tri.simplices:
        x, y, z = edge
        d1 = math.sqrt(pow(points[x][0] - points[y][0],2)+pow(points[x][1] - points[y][1],2))
        d2 = math.sqrt(pow(points[y][0] - points[z][0],2)+pow(points[y][1] - points[z][1],2))
        d3 = math.sqrt(pow(points[x][0] - points[z][0],2)+pow(points[x][1] - points[z][1],2))
        # graph.add_weighted_edges_from([(x,y),(y,z),(x,z)])
        graph.add_edge(x,y)
        graph.add_edge(y,z)
        graph.add_edge(x,z)
    return G

def read_txt(url,vanish_labels):
    pts = []
    vanish_pts = []
    with open(url, 'r') as file:
        # 逐行读取文件内容
        for i, line in enumerate(file, start=1):
            # 去除行末的换行符
            line = line.strip()
            # 切分行内容，以空格分割
            parts = line.split(':')
            # 第一个元素是i
            i_value = int(parts[0])
            # 括号中的内容
            bracket_content = parts[1][1:-1]  # 去除括号
            # 括号中的内容切分，以逗号分割
            nums = bracket_content.split(',')
            x = int(nums[0])
            y = int(nums[1])
            # 解析后的内容
            # print("index:{},coordinates:{}".format(i_value,(x,y)))
            if i_value not in vanish_labels:
                pts.append((i_value,x,y))
            else:
                vanish_pts.append((i_value,x,y))
    return np.array(pts),vanish_pts


def normalize_2d(src_points):
    src_ax,src_ay = np.mean(src_points,axis=0)
    n = len(src_points)
    ## 平移矩阵
    T1 = np.array([[1,0,-src_ax],
                [0,1,-src_ay],
                [0,0,1]])
    p1 = (T1 @ np.array([[x,y,1] for x,y in src_points]).T).T
    src_sum = 0
    for i in range(n):
        src_sum += math.sqrt(p1[i][0]**2+p1[i][1]**2)
    src_sum /= n
    alpha = math.sqrt(2) / src_sum
    ## 缩放矩阵
    S1 = np.array([[alpha,0,0],
                [0,alpha,0],
                [0,0,1]])
    u = (S1 @ p1.T).T
    return u[:,:2], S1 @ T1

def add_gaussian_noise(points, mean, std_dev):
    noise = np.random.normal(mean, std_dev, size=(len(points), 2))
    noisy_points = np.array(points) + noise
    return noisy_points
def generate_points(num_points, min_distance, max_coordinate, vanish_labels,angle,seed=None,var=1):
    if seed is not None:
        random.seed(seed)
    points = []
    vanish_points = []
    idx = 1
    noise = np.random.normal(0, var, size=(num_points, 2))
    while len(points) < num_points:
        x = random.randint(0, max_coordinate)
        y = random.randint(0, max_coordinate)
        new_point = (x, y)
        if check_distance(points, new_point, min_distance):
            x,y = (x,y) + noise[idx-1]
            if idx in vanish_labels:
                vanish_points.append([idx,x,y])
            points.append([idx,x,y])
            idx += 1
    for item in vanish_points:
        points.remove(item)
    

    rotation_matrix = np.array([[math.cos(math.radians(angle)), math.sin(math.radians(angle))],
                                [-math.sin(math.radians(angle)), math.cos(math.radians(angle))]])
    points = np.array(points)
    pts_tmp = points[:,1:].copy()
    rotated_points = np.dot(pts_tmp, rotation_matrix).astype(np.int)
    points[:,1:] = rotated_points

    if len(vanish_points) != 0:
        vanish_points = np.array(vanish_points)
        van_pts_tmp = vanish_points[:,1:].copy()
        rotated_vanish_points = np.dot(van_pts_tmp, rotation_matrix).astype(np.int)
        vanish_points[:,1:] = van_pts_tmp
    return points, vanish_points

def check_distance(points, new_point, min_distance):
    for point in points:
        if calculate_distance(point, new_point) < min_distance:
            return False
    return True

def calculate_distance(point1, point2):
    x1, y1 = point1[1:]
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def plot_points(points, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], color='blue')
        plt.text(point[0], point[1], str(i+1), fontsize=10, ha='left', va='bottom')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original Distribution')
    plt.grid(True)

    # 逆时针旋转45°的旋转矩阵
    rotation_matrix = np.array([[math.cos(math.radians(45)), math.sin(math.radians(45))],
                                [-math.sin(math.radians(45)), math.cos(math.radians(45))]])

    # 将坐标点转换为矩阵
    points_matrix = np.array(points)

    # 应用旋转矩阵
    rotated_points = np.dot(points_matrix, rotation_matrix)

    plt.subplot(1, 2, 2)
    for i, point in enumerate(rotated_points):
        plt.scatter(point[0], point[1], color='red')
        plt.text(point[0], point[1], str(i+1), fontsize=10, ha='left', va='bottom')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rotated Distribution')
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 设置随机种子
# seed = 607
# num_points = 30
# min_distance = 30
# max_coordinate = 800
# angle = 45

# random_points = generate_points(num_points, min_distance, max_coordinate, angle, seed)
# plot_points(random_points, title='Randomly Generated Points')