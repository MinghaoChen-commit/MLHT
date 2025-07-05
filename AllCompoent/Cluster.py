from tools.settings import *

def DBSCAN_Anaylse(data):
    features = np.array([[x + w/2, y + h/2] for id, x, y, w, h, frame in data])
    frames = np.array([frame for id, x, y, w, h, frame in data])

    # 计算距离矩阵，同时考虑帧编号的限制
    def compute_distance_matrix(features, frames):
        n = len(features)
        distance_matrix = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                if frames[i] != frames[j]:
                    # if(abs(int( frames[i]) - int(frames[j])) <= 3):
                        distance = np.linalg.norm(features[i] - features[j])
                        distance_matrix[i, j] = distance
                        distance_matrix[j, i] = distance
        return distance_matrix

    # 替换无穷大值为系统中最大的浮点数
    def replace_inf_values(distance_matrix):
        max_float = np.finfo(np.float64).max
        distance_matrix[np.isinf(distance_matrix)] = max_float
        return distance_matrix

    # 使用DBSCAN进行初始聚类
    def perform_dbscan(distance_matrix, eps=50, min_samples=1):
        return DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix).labels_

    # 使用AgglomerativeClustering进行子簇聚类
    def perform_agglomerative_clustering(features, n_clusters):
        return AgglomerativeClustering(n_clusters=n_clusters).fit(features).labels_

    # 检查并拆分簇，直到每个簇中没有相同帧的目标或达到递归次数上限
    def check_and_split_clusters(data, features, frames, labels, max_recursion_depth=5, current_recursion_depth=0, eps=50, min_samples=1):
        if current_recursion_depth > max_recursion_depth:
            return labels

        new_labels = np.full(len(labels), -1)  # 初始化新标签数组
        unique_labels = np.unique(labels)
        current_label = 0

        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # 跳过噪声点
            cluster_indices = np.where(labels == cluster_id)[0]
            frames_in_cluster = frames[cluster_indices]

            if len(set(frames_in_cluster)) == len(frames_in_cluster):
                # No duplicate frames in this cluster
                new_labels[cluster_indices] = current_label
                current_label += 1
            else:
                # Duplicate frames found, further split the cluster
                sub_features = features[cluster_indices]
                sub_frames = frames[cluster_indices]

                n_sub_clusters = len(set(sub_frames))
                sub_labels = perform_agglomerative_clustering(sub_features, n_clusters=n_sub_clusters)

                # 递归检查和拆分子簇
                sub_labels = check_and_split_clusters(
                    data[cluster_indices], sub_features, sub_frames, sub_labels,
                    max_recursion_depth, current_recursion_depth + 1, eps, min_samples
                )

                for sub_cluster_id in np.unique(sub_labels):
                    if sub_cluster_id == -1:
                        continue  # 跳过子簇中的噪声点
                    sub_cluster_indices = np.where(sub_labels == sub_cluster_id)[0]
                    new_labels[cluster_indices[sub_cluster_indices]] = current_label
                    current_label += 1

        return new_labels

    # 手动分配相同帧目标到不同簇
    def assign_unique_frames_to_clusters(data, labels):
        frames = [item[-1] for item in data]
        unique_labels = np.unique(labels)
        final_labels = np.full(len(labels), -1)
        current_label = 0

        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue
            cluster_indices = np.where(labels == cluster_id)[0]
            frames_in_cluster = [frames[i] for i in cluster_indices]

            frame_dict = {}
            for idx in cluster_indices:
                frame = frames[idx]
                if frame not in frame_dict:
                    frame_dict[frame] = current_label
                    final_labels[idx] = current_label
                    current_label += 1
                else:
                    final_labels[idx] = frame_dict[frame]

        return final_labels

    # 初始距离矩阵计算和替换无穷大值
    distance_matrix = compute_distance_matrix(features, frames)
    distance_matrix = replace_inf_values(distance_matrix)

    # 初次聚类
    initial_labels = perform_dbscan(distance_matrix)

    unique_final_labels = np.unique(initial_labels)
    for cluster_id in unique_final_labels:
        cluster_indices = np.where(initial_labels == cluster_id)[0]
        print(f"Cluster {cluster_id}:")
        for idx in cluster_indices:
            print(data[idx])

    # 对聚类结果进行检查和处理，设置递归次数上限为3
    max_recursion_depth = 3
    new_labels = check_and_split_clusters(np.array(data), features, frames, initial_labels, max_recursion_depth)

    # # 手动分配相同帧目标到不同簇
    # final_labels = assign_unique_frames_to_clusters(data, new_labels)


    result = []
    # 打印每个簇中的目标框
    unique_final_labels = np.unique(new_labels)
    with open('juleifexin.txt', "w", encoding="utf-8") as file:
        for cluster_id in unique_final_labels:
            cluster_indices = np.where(new_labels == cluster_id)[0]
            branch={
                "cluster_id": cluster_id ,
                "targetsid": []
            }
            print(f"Cluster {cluster_id}:")
            file.write(f"Cluster {cluster_id}:"+"\n")
            for idx in cluster_indices:
                # 写入内容
                file.write(str(data[idx])+"\n")
                print(data[idx])
                branch["targetsid"].append(data[idx])
            result.append(branch)

    return result

def CreateNewBranch(all_data_GMOT,branch):
    NewBranch = []
    for target in branch:
        id = target[0]
        frame = target[5]
        for Everyframe in all_data_GMOT:
            if frame == Everyframe["frame"]:
                for target in Everyframe["targets"]:
                    if id == target.id:
                        NewBranch.append(target)
    trajectory_branch = Trajectory(NewBranch,GetBranchID())
    return trajectory_branch

def check_last_element_duplicates(matrix):
    seen = set()
    for sublist in matrix:
        last_element = sublist[-1]
        if last_element in seen:
            return True
        seen.add(last_element)
    return False

def BuildTreeFromDBSCAN(results,all_data_GMOT,Froest):
    for result in results:
            branch =  result["targetsid"]
            if(check_last_element_duplicates(branch) == False):
                 Newbranch = CreateNewBranch(all_data_GMOT,branch)
                 Froest.append(Newbranch)
            else:
                print(branch)

