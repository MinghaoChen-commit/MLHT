
import os
from shutil import rmtree

import numpy as np

from scipy.spatial.distance import cdist

import uuid
import hashlib
import random

import cv2

from tools.DataStructureClass import Target , Trajectory, VisualizeStructure
from tools.PrepareData import config
from natsort import natsorted

def Custom_Euclidean(u, v):
    distance = np.linalg.norm(u[0:2] - v[0:2])
    if distance > max(u[2], v[2]):
        distance = 0
    else:
        distance = 1
    return distance

def Custom_Euclidean_Match(u, v):
    distance = np.linalg.norm(u[1:3] - v[1:3])
    return distance

def Custom_Cosine(u, v):
    # 计算向量 u 和 v 的点积
    dot_product = np.dot(u, v)
    
    # 计算向量 u 和 v 的范数（模）
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    
    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_u * norm_v)
    
    return cosine_similarity



def GetBranchID():
    return str(uuid.uuid1())


def VisSet(set):
    idsL = []
    idsR = []
    for index,branch in enumerate(set):
        idsL = [target.sid for target in branch[0].targets]
        idsR = [target.sid for target in branch[1].targets]
        # if("0-4" in idsL or "0-4" in idsR):          
        print(str(index) + "\t0:" +str(idsL))
        print(str(index) + "\t1:" +str(idsR) + "\n") 

def VisID(forest,Curframe, frame = ""):

    for index, branch in enumerate(forest):
        if type(branch) == dict:
            ids = [target.sid for target in branch["targets"]]
        else:
            ids = [target.sid for target in branch.targets]
        
        gtids = list(set(['#'+str(target.gtid) for target in branch.targets]))

        print("[" + str(index) + "]", "[" + str(len(ids)) + "]", "[" + "{:.4f}".format(branch.GetTrajectoryWeight_DIS_Continue(Curframe)[0]) + "]", \
               str(ids), str(gtids))
        # print()
    
        # for target1 in branch.targets:
        #     for target2 in branch.targets:
                
        #         print(str(index)+"\t"+str(target1.id)+"\t"+str(target2.id)+"\t"+str(np.dot(target1.appearance, target2.appearance)))
    

    print("[" + str(frame) + "]" + "\t" + "Total Branch: ", len(forest))

def GetSimilarityMatrix(sources, targets):
    if len(sources) * len(targets) == 0:
        return []
    
    sources_appearance = np.array([branch.targets[-1].appearance for branch in sources])    
    targets_appearance = np.array([target.appearance for target in targets])
    similarity_appearance = cdist(sources_appearance, targets_appearance, metric=Custom_Cosine)

    sources_position = np.array([branch.targets[-1].bbox for branch in sources])    
    targets_position = np.array([target.bbox for target in targets])
    similarity_position = cdist(sources_position, targets_position, metric=Custom_Euclidean)

    similarity = np.multiply(similarity_appearance, similarity_position)

    return similarity




#根据branch_id生成颜色
def GetColor(id):
    hash_val = hashlib.md5(id.encode()).hexdigest()
    
    # 将16进制哈希值转换为整数
    hash_int = int(hash_val, 16)
    
    # 使用随机种子确保每次运行生成的颜色一致
    random.seed(hash_int)
    
    # 生成随机的RGB颜色值
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    return (r, g, b)


def visualize(final_trajectory):
    # 批量读取所有需要处理的图像
    image_dict = []
    Evaluation_list=[]

    imageList = os.listdir(config['ORIGIN_PATH'])

    for index, image in enumerate(imageList):
        origin_path = os.path.join(config['ORIGIN_PATH'], image)
        image_dict.append({"name": image, "image": cv2.imread(origin_path)})           
    
    # 批量处理图像并保存修改后的图像
    for index, trajectory in enumerate(final_trajectory):
        color_for_bbox = GetColor(trajectory.targets[0].sid.replace("-", ""))  # 使用第一个目标的sid获取颜色
        
        for target in trajectory.targets:
            # 获取图像
            image = image_dict[target.frame-1]["image"]
            
            x_lu = int(target.x - 0.0 * target.width)
            y_lu = int(target.y - 0.0 * target.height)
            x_rl = int(target.x + 1.0 * target.width)
            y_rl = int(target.y + 1.0 * target.height)

            # 绘制边界框和文本
            cv2.rectangle(image, (x_lu, y_lu), (x_rl, y_rl), color_for_bbox, thickness=config['BOUNDINGBOX_THICKNESS'])
            cv2.putText(image, str(target.id), (x_lu, y_lu + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color_for_bbox, config['BOUNDINGBOX_THICKNESS'] - 1)

            # 保存Results
            Evaluation_list.append([target.frame, index + 1, target.x, target.y, target.width, target.height, target.confidence, -1, -1])
    
    if os.path.exists(config['VISUALIZE_PATH']):
        rmtree(config['VISUALIZE_PATH'])
    
    os.makedirs(config['VISUALIZE_PATH'])

    for index, image in enumerate(image_dict):
        content_name = image["name"]
        modified_path = os.path.join(config['VISUALIZE_PATH'], content_name)
        cv2.imwrite(modified_path, image["image"])
    
    sorted_lists = sorted(Evaluation_list, key=lambda x: x[0])

    # 将排序后的列表转换为逗号分隔的字符串并写入文本文件
    with open('res.txt', 'w') as f:
        for sublist in sorted_lists:
            line = ','.join(map(str, sublist))
            f.write(line + '\n')


def visualizeGMOT(final_trajectory, all_data_DJI):
    FrameOriginzation={}
    for frame in range(int(all_data_DJI[0]["frame"]),int(all_data_DJI[-1]["frame"]+1)):
        FrameOriginzation[frame] = []
    
    for branch in final_trajectory:
        for target in branch.targets:
            FrameOriginzation[target.frame].append(VisualizeStructure(target,branch))
    
    for index, frame in enumerate(FrameOriginzation):
        if(FrameOriginzation[frame] != None):
            PicturePath="D:\DJI\img/"+str(frame + 1).zfill(6)+".jpg"
            img = cv2.imread(PicturePath)
            for EveryBox in FrameOriginzation[frame]:
                x_lu = int(EveryBox.x - 0.5 * EveryBox.w)
                y_lu = int(EveryBox.y - 0.5 * EveryBox.h)
                x_rl = int(EveryBox.x + 0.5 * EveryBox.w)
                y_rl = int(EveryBox.y + 0.5 * EveryBox.h)

                # 绘制边界框和文本
                cv2.rectangle(img, (x_lu, y_lu), (x_rl, y_rl), EveryBox.color_for_bbox, thickness=config['BOUNDINGBOX_THICKNESS'])
                cv2.putText(img, str(EveryBox.id), (x_lu, y_lu + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, EveryBox.color_for_bbox, config['BOUNDINGBOX_THICKNESS'] - 1)

            OutputPicturePath=config["ROOT_VIDEO_RES"]+str(frame).zfill(6)+".jpg"
            cv2.imwrite(OutputPicturePath, img)
            # print("success")

#保存结果
def SavetheResult(final_trajectory):
    Evaluation_list=[]
    for index, trajectory in enumerate(final_trajectory):
        for target in trajectory.targets:     
            # 保存Results
            Evaluation_list.append([target.frame, index + 1, target.x, target.y, target.width, target.height, 1, -1, -1, -1])
    
    sorted_lists = sorted(Evaluation_list, key=lambda x: x[0])

    with open(config["MOTA_RES"], 'w') as f:
        for sublist in sorted_lists:
            line = ','.join(map(str, sublist))
            f.write(line + '\n')

    return True

#合成视频
def images_to_video(image_folder, output_video, frame_rate):
    # 获取图片列表并按自然顺序排序
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images = natsorted(images)

    # 读取第一张图片以获取帧的尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 定义视频编码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # 逐帧写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放 VideoWriter 对象
    video.release()

def VideoBuilder():
    images_to_video(config["VISUALIZE_PATH"],config["VIDEO_RES"],config["VIDEO_FRAME"])



def regenerateGraphwithFrame(targets):
    Graph=[]
    for target in targets:
        id = target.id
        x = target.x + 0.5*target.width
        y = target.y + 0.5*target.height
        Graph.append(np.array([id, x, y]))

    return np.array(Graph)

def regenerateGraphwithForest(trajectory_forest, graphMatchList, frame):
    Graph=[]
    id_list = [tup[0] for tup in graphMatchList]
    for branchs in trajectory_forest:
        target = branchs.targets[-1]
        id = target.id
        if id in id_list or frame - target.frame > 1:
            continue
        x = target.x + 0.5 * target.width
        y = target.y + 0.5 * target.height
        Graph.append(np.array([id, x, y]))

    unique_lst = []
    if len(Graph) > 0:
        # 将NumPy数组转换为元组并去除重复项
        unique_tuples = {tuple(arr) for arr in Graph}
        
        # 将元组转换回NumPy数组构成新的列表
        unique_lst = [np.array(tup) for tup in unique_tuples]

    return np.array(unique_lst)