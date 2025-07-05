from tools.settings import *

def regenerateGraphWithThreshold(graph_current, graph_previous):
    n1 = len(graph_current)
    n2 = len(graph_previous)
   
    if n1 * n2 is not 0:
        position_current = np.array([node for node in graph_current])    
        position_previous = np.array([node for node in graph_previous])
        distance_matrix = cdist(position_current, position_previous, metric = Custom_Euclidean_Match)
 
        # 找到每行最小的两个值
        sorted_rows = np.sort(distance_matrix, axis=1)
        row_min_elements = sorted_rows[:, :2].tolist()
        
        # 找到每列最小的两个值
        sorted_cols = np.sort(distance_matrix, axis=0)
        col_min_elements = sorted_cols[:2, :].T.tolist()

        graph_current = np.hstack((graph_current, row_min_elements))
        graph_previous = np.hstack((graph_previous, col_min_elements))

    return graph_current, graph_previous


#图匹配
def SupportMatrixMatchV2(GraphinBranch, GraphSolitude, parameters):

    start_time = time.time()

    alpha = parameters["alpha"]
    dist_errorthredhold = parameters["dist_errorthredhold"]
    iouthredhold = parameters["iouthredhold"]
    SupportMaxrixthredhold = parameters["SupportMaxrixthredhold"]
    lowThresh = parameters["Sequence_LowThredholdForOneStage"]
    highThresh = parameters["Sequence_HighThredholdForOneStage"]

    left_points = GraphinBranch
    right_points = GraphSolitude
    prior_hypothesis = []

    left_graph = init_graph(left_points)
    right_graph = init_graph(right_points)

    end_time = time.time()
    #print("#####", end_time-start_time)
    start_time = time.time()

    triPolo_simi_matrix = calculate_tri_simi(left_graph = left_graph, left_pts = left_points[:, 1:3], \
                                             right_graph = right_graph, right_pts = right_points[:, 1:3])
    if(np.all(triPolo_simi_matrix== 0)):
        return []
    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()
    
    left_adj_matrix = nx.adjacency_matrix(left_graph['graph']).toarray()
    right_adj_matrix = nx.adjacency_matrix(right_graph['graph']).toarray()

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    left_entropy = calculate_adjacency_entropy(left_graph['graph'])
    right_entropy = calculate_adjacency_entropy(right_graph['graph'])

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    left_betw = nx.closeness_centrality(left_graph['graph'])
    right_betw = nx.closeness_centrality(right_graph['graph'])

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    left_friends = dict()
    right_friends = dict()
    for i in range(len(left_points)):
        left_friends[i] = []
    for i in range(len(right_points)):
        right_friends[i] = []
    find_friends(left_friends, left_adj_matrix, left_graph, \
                 right_friends,right_adj_matrix,right_graph)
    # support_simi_matrix = calculate_support_simi(left_friends,right_friends)
    
    n1 = len(left_points)
    n2 = len(right_points)

    ## 相似度矩阵计算
    adjEntro_simi_matrix = np.zeros((n1, n2)).astype(np.float32)
    max_value = 0
    min_value = 10
    left_key = dict()
    right_key = dict()


    for i in range(n1):
        left_key[i] = alpha * left_entropy[i] + (1 - alpha) * math.exp(left_betw[i])
    for i in range(n2):
        right_key[i] = alpha * right_entropy[i] + (1 - alpha) * math.exp(right_betw[i])

    for i in range(adjEntro_simi_matrix.shape[0]):
        for j in range(adjEntro_simi_matrix.shape[1]):
            adjEntro_simi_matrix[i,j] = math.cos(np.linalg.norm(left_key[i]-right_key[j]))

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    simi_matrix = np.zeros_like(adjEntro_simi_matrix)
    # simi_matrix = support_simi_matrix + triPolo_simi_matrix
    simi_matrix = triPolo_simi_matrix
    matches = find_best_matches(simi_matrix,prior_hypothesis)
    one_stage_matches = None
    two_stage_matches = None
    H = None
    if len(matches) >= 4:
        train_pts = []
        query_pts = []
        index_pts = []
        for i in range(len(matches)):
            train_pts.append(left_points[int(matches[i][0])])
            query_pts.append(right_points[int(matches[i][1])])
            index_pts.append(matches[i])
        train_pts = np.array(train_pts)
        query_pts = np.array(query_pts)
        # 计算匹配的单应性误差
        matches_index = []
        matches_labels = []
        # 坐标归一化，平移
        left_ax,left_ay = np.mean(left_points[:, 1:3], axis=0).astype(int)
        right_ax,right_ay = np.mean(right_points[:, 1:3], axis=0).astype(int)
        left_points[:, 1:3] -= (left_ax,left_ay)
        right_points[:, 1:3] -= (right_ax,right_ay)

        for pair in matches:
            i = int(pair[0])
            j = int(pair[1])
            matches_index.append([left_graph['index'][left_points[i][0]],right_graph['index'][right_points[j][0]]])
            matches_labels.append([left_points[i][0],right_points[j][0]])
        one_stage_matches, err_match, H = dual_threshing(left_points, left_adj_matrix, left_entropy, left_graph, \
                                                        right_points, right_adj_matrix, right_entropy, right_graph, \
                                                        matches_index, matches_labels, \
                                                        lowThresh, highThresh)
    else:
        return matches
    
    if H is None:
        two_stage_matches = []
        for item in matches:
            two_stage_matches.append((left_graph['labels'][item[0]],right_graph['labels'][item[1]]))
        return two_stage_matches

    one_stage_matches_index = []
    for item in one_stage_matches:
        i = left_graph['index'][item[0]]
        j = right_graph['index'][item[1]]
        one_stage_matches_index.append((i,j))
    one_stage_H = H

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    left_error = []
    right_error = []
    for x in err_match:
        if x[0] != -1:
            left_error.append(left_graph['index'][x[0]])
        if x[1] != -1:
            right_error.append(right_graph['index'][x[1]])
    two_stage_matches = []

    dist_error_matrix = np.zeros((len(left_error),len(right_error)),dtype=int)

    for i,l_index in enumerate(left_error):
        for j, r_index in enumerate(right_error):
            trans_right_pts = H @ [left_points[l_index][1], left_points[l_index][2], 1]
            trans_right_pts = (trans_right_pts / trans_right_pts[2])[:2]
            trans_left_pts = np.linalg.inv(H) @ [right_points[r_index][1], right_points[r_index][2], 1]
            trans_left_pts = (trans_left_pts / trans_left_pts[2])[:2]
            sd = np.linalg.norm(trans_left_pts - left_points[l_index][1:3])
            dd = np.linalg.norm(trans_right_pts - right_points[r_index][1:3])
            dist_error_matrix[i,j] = sd + dd

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    row_ind,col_ind = linear_sum_assignment(dist_error_matrix,False)
    left_bad_support_index = []
    right_bad_support_index = []
    matches_set = set(one_stage_matches)

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    for i, j in zip(row_ind, col_ind):
        l_index = left_error[i]
        r_index = right_error[j]
        a = left_graph['labels'][l_index]
        b = right_graph['labels'][r_index]
        iou = calculate_second_IoU(l_index,left_friends,left_graph,r_index,right_friends,right_graph,matches_set,0.4)
        if dist_error_matrix[i,j] < 10 or (dist_error_matrix[i,j] < dist_errorthredhold and iou > iouthredhold): 
            two_stage_matches.append((left_graph['labels'][l_index],right_graph['labels'][r_index]))
        else:
            left_bad_support_index.append(l_index)
            right_bad_support_index.append(r_index)

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    two_stage_matches += one_stage_matches

    support_matrix = np.zeros((len(left_bad_support_index),len(right_bad_support_index)),dtype=float)
    matches_set = set(two_stage_matches)
    ## 二阶支持度计算
    for i, l_index in enumerate(left_bad_support_index):
        for j, r_index in enumerate(right_bad_support_index):
            a = left_graph['labels'][l_index]
            b = right_graph['labels'][r_index]
            iou = calculate_second_IoU(l_index,left_friends,left_graph,r_index,right_friends,right_graph,matches_set,0.4)
            # print("{}---{},IoU:{}".format(a,b,iou))
            support_matrix[i,j] = iou

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    matches = find_best_matches(support_matrix,None)
    for item in matches:
        i,j = item
        if support_matrix[i,j] > 0.6:
            two_stage_matches.append((left_graph['labels'][left_bad_support_index[i]],right_graph['labels'][right_bad_support_index[j]]))
    
    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    two_stage_matches_index = []
    for item in two_stage_matches:
        two_stage_matches_index.append((left_graph['index'][item[0]],right_graph['index'][item[1]]))
    ans = 0
    H = RHO_homography(left_points, right_points, two_stage_matches_index)
    
    if H is None:
        two_stage_matches = []
        for item in matches:
            two_stage_matches.append((left_graph['labels'][item[0]],right_graph['labels'][item[1]]))
        return two_stage_matches

    two_bad_pairs = []
    two_left_bad_index = []
    two_right_bad_index = []
    sd = 0
    dd = 0
    for i, item in enumerate(two_stage_matches_index):
        hd_pts = H @ [left_points[item[0]][1], left_points[item[0]][2], 1]
        hd_pts = (hd_pts / hd_pts[2])[:2]
        
        hs_pts = np.linalg.inv(H) @ [right_points[item[1]][1], right_points[item[1]][2], 1]
        hs_pts = (hs_pts / hs_pts[2])[:2]
        
        sd = np.linalg.norm(hs_pts - left_points[two_stage_matches_index[i][0]][1:3])
        dd = np.linalg.norm(hd_pts - right_points[two_stage_matches_index[i][1]][1:3])
        gd = sd + dd 
        if gd > 40:
            two_bad_pairs.append(two_stage_matches_index[i])
            two_left_bad_index.append(two_stage_matches_index[i][0])
            two_right_bad_index.append(two_stage_matches_index[i][1])
    for item in two_bad_pairs:
        two_stage_matches_index.remove(item)

    end_time = time.time()
    #print(end_time-start_time)
    start_time = time.time()

    re_dist = np.zeros((len(two_left_bad_index),len(two_right_bad_index)))
    for i, l_index in enumerate(two_left_bad_index):
        for j, r_index in enumerate(two_right_bad_index):
            trans_right_pts = H @ [left_points[l_index][1], left_points[l_index][2], 1]
            trans_right_pts = (trans_right_pts / trans_right_pts[2])[:2]
            trans_left_pts = np.linalg.inv(H) @ [right_points[r_index][1], right_points[r_index][2], 1]
            trans_left_pts = (trans_left_pts / trans_left_pts[2])[:2]
            sd = np.linalg.norm(trans_left_pts - left_points[l_index][1:3])
            dd = np.linalg.norm(trans_right_pts - right_points[r_index][1:3])
            re_dist[i,j] = sd + dd
    row_ind,col_ind = linear_sum_assignment(re_dist,False)
    for i, j in zip(row_ind, col_ind):
        l_index = two_left_bad_index[i]
        r_index = two_right_bad_index[j]
        two_stage_matches_index.append((l_index,r_index))
    two_stage_matches.clear()
    for item in two_stage_matches_index:
        two_stage_matches.append((left_graph['labels'][item[0]],right_graph['labels'][item[1]]))

    end_time = time.time()
    #print(end_time-start_time)

    return two_stage_matches
