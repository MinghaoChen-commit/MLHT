from Graphmatch.Entity import Item
import cv2
import numpy as np
from Graphmatch.utilsformatch import *
from scipy.optimize import least_squares

def RHO_homography(left_points,right_points,match_index):
    src_pts = []
    dst_pts = []
    H = None
    for item in match_index:
        src_pts.append(left_points[item[0]][1:3])
        dst_pts.append(right_points[item[1]][1:3])
    src_pts = np.array(src_pts).reshape(-1,1,2)
    dst_pts = np.array(dst_pts).reshape(-1,1,2)
    # cv2.RHO算法优于cv2.FM_RANSAC
    if len(match_index) >= 4:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO, 4.0)
    return H
    


def calculate_ls(left_points,left_weight,right_points,right_weight,match_index):
    left_normal_pts, LT = normalize_2d(np.array(left_points[:,1:3]))
    right_normal_pts, RT = normalize_2d(np.array(right_points[:,1:3]))
    A = []
    for pair in match_index:
        x1,y1 = left_normal_pts[pair[0]]
        x2,y2 = left_normal_pts[pair[0]]
        A.append((x1,y1,1,0,0,0,-x1*x2,-y1*x2,-x2))
        A.append((0,0,0,x1,y1,1,-x1*y2,-y1*y2,-y2))
    np.array(A)
    U,Sigma,VT = np.linalg.svd(A)
    H = VT.T[:,-1].reshape(-1,3)
    H_initial = np.linalg.inv(RT) @ H @ LT
    result = least_squares(homography_error, H_initial.flatten(), args = (left_points, right_points, match_index))
    H_optimized = result.x.reshape((3, 3))
    H = H_optimized / H_optimized[2,2]
    return H

def homography_error(params, left_points, right_points, index):
    H = params.reshape((3, 3))  # 重构H矩阵
    error = []
    for item in index:
        homo_src_pts = np.array([left_points[item[0]][1], left_points[item[0]][2], 1])
        homo_dst_pts = np.array([right_points[item[1]][1], right_points[item[1]][2], 1])
        
        # 直接变换
        hd_pts = H @ homo_src_pts
        hd_pts = (hd_pts / hd_pts[2])[:2]
        
        # 逆变换
        hs_pts = np.linalg.inv(H) @ homo_dst_pts
        hs_pts = (hs_pts / hs_pts[2])[:2]
        
        sd = np.linalg.norm(hs_pts - left_points[item[0]][1:3])
        dd = np.linalg.norm(hd_pts - right_points[item[1]][1:3])
        
        error.append(sd)
        error.append(dd)
    return np.array(error)

def calculate_error(H, left_points, right_points, index):
    homo_src_pts = []
    homo_dst_pts = []
    error = []
    for item in index:
        homo_src_pts.append([left_points[item[0]][1], left_points[item[0]][2], 1])
        homo_dst_pts.append([right_points[item[1]][1], right_points[item[1]][2], 1])
    sd = 0
    dd = 0
    for i in range(len(index)):
        hd_pts = H @ homo_src_pts[i]
        hd_pts = (hd_pts / hd_pts[2])[:2]
        
        hs_pts = np.linalg.inv(H) @ homo_dst_pts[i]
        hs_pts = (hs_pts / hs_pts[2])[:2]
        
        sd = np.linalg.norm(hs_pts - left_points[index[i][0]][1:3])
        dd = np.linalg.norm(hd_pts - right_points[index[i][1]][1:3])
        error.append(sd)
        error.append(dd)
    return error
    
def point_to_epiline_distance(point, line):
    return np.abs(line[0]*point[0] + line[1]*point[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)
def estimate_epiline_error(F,left_vanish_pts,right_vanish_pts):
    distances = dict()
    for i,(pt1, pt2) in enumerate(zip(left_vanish_pts, right_vanish_pts)):
        epiline = np.dot(F, np.array([pt1[0], pt1[1], 1]))
        distance = point_to_epiline_distance(pt2, epiline)
        pair = (left_vanish_pts[i][0],right_vanish_pts[i][0])
        distances[pair] = distance
    return distances
def predict(left_pts,left_vanish_pts,right_pts,right_vanish_pts,matches):
    left_labels = [x[0] for x in left_pts]
    right_labels = [x[0] for x in right_pts]
    left_vanish_pts_tmp = left_vanish_pts.copy()
    right_vanish_pts_tmp = right_vanish_pts.copy()
    left_label2index = dict()
    right_label2index = dict()
    for i in range(len(left_labels)):
        left_label2index[left_labels[i]] = i
    for i in range(len(right_labels)):
        right_label2index[right_labels[i]] = i

    for vanish_pt in right_vanish_pts_tmp:
        index = left_label2index[vanish_pt[0]]
        left_vanish_pts.append(left_pts[index])
    for vanish_pt in left_vanish_pts_tmp:
        index = right_label2index[vanish_pt[0]]
        right_vanish_pts.append(right_pts[index])

    left_vanish_pts.sort(key = lambda x : x[0])
    right_vanish_pts.sort(key = lambda x : x[0])
    left_vanish_pts = np.array(left_vanish_pts)
    right_vanish_pts = np.array(right_vanish_pts)

    src_pts = []
    dst_pts = []
    for item in matches:
        src_pts.append(left_pts[item[0]][1:3])
        dst_pts.append(right_pts[item[1]][1:3])
    src_pts = np.array(src_pts).reshape(-1,2)
    dst_pts = np.array(dst_pts).reshape(-1,2)
    F, _ = cv2.findFundamentalMat(src_pts, dst_pts, method=cv2.FM_RANSAC,ransacReprojThreshold=0.9, confidence=0.99)
    dist = estimate_epiline_error(F,left_vanish_pts,right_vanish_pts)
    return dist

    