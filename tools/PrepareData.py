
import os
import csv
import pandas as pd
import numpy as np
import cv2

from tools.DataStructureClass import Target , Trajectory, Nstr

config = {
    # "DATAINPUTPATH":"D://PaperForReId/GMOT40/GMOT40/track_label/boat-2.txt",
    # "DATAOUTPUTPATH":"D://PaperForReId/GMOT40/detectionBoat-2-gt.csv",

    "DATAINPUTPATH":"D://PaperForReId/GMOT40/GMOT40/detection_label/boat-2.txt",
    "DATAOUTPUTPATH":"D://PaperForReId/GMOT40/detection/Boat-2-det.csv",

    "DJIDATAPIC":"D://DJI/person_dji/datasets/train/images",
    # "DJIDATALABEL":


    "LABELROOTPATH": "D:\PaperForReId\GMOT40\GMOT40/track_label",
    "PICTUREROOTPATH":"D:\PaperForReId\GMOT40\GMOT40\GenericMOT_JPEG_Sequence",
    "CSVROOTPATH":"D:\PaperForReId\GMOT40\GMOT40\TempCSV/",
    # "ROOT_MOTA_RES":'./ResForTXTandVideo/',
    "ROOT_MOTA_RES":'D:/PaperForReId/MOTChallengeEvalKit-master/MOTChallengeEvalKit-master/res/GMOTres-sub',
    "ROOT_VIDEO_RES":"./ResForTXTandVideo/",


    "SAMPLE_PATH":"D://PaperForReId/DJIdataset/DJI_20240514120258_0001_D/000001.jpg",
        
    "APPEARANCE_SIMILARITY_THRESHOLD": 0.9,
    "DISTANCE_WEIGHT": 0.2,
    "APPEARANCE_WEIGHT": 0.8,
    "MAX_FRAME_GAP": 5,
    "MAX_DIATANCE_GAP": 0.95,

    # "BRANCH_COUNT_THRESHOLD": 90,
    "BRANCH_COUNT_THRESHOLD": 100,
    "DETECTION_CONFIDENCE": 0.6,
    'VISUALIZE_PATH':'./Visualize/',
    'BOUNDINGBOX_THICKNESS': 5,
    'THREDHOLD_TRACKLET_MIN_LENGTH': 3,
    'THREDHOLD_TRACKLET_MAX_LENGTH': 5,
    'THREDHOLD_FRAME_INTERVAL': 3,
    'READ_NUMBER_Frame': 9,
    'TRACKLET_REPEAT_TIMES':2,

    'ORIGIN_PATH':'D://PaperForReId/GMOT40/GMOT40/GenericMOT_JPEG_Sequence/boat-2/img1/',
    
    
    "TOTAL_START":0,
    "TOTAL_END":20,

    "MOTA_RES":'./ResForTXTandVideo/ResGMOT.txt',
    "VIDEO_RES":"./ResForTXTandVideo/GMOT-Boat2.mp4",
    "VIDEO_FRAME": 30,
    
    "MATCH_RESULT": "./GraphMatchkpl/GraphMatch.pkl",
    
    "ReMatch_Thredhold": 0.3,

    "Sequence_alpha" : [0.6, 0.7, 0.8, 0.9],
    "Sequence_dist_errorthredhold" : [60, 70, 80, 90],
    "Sequence_iouthredhold" : [0.6, 0.7, 0.8, 0.9],
    "Sequence_SupportMaxrixthredhold" : [0.6, 0.7, 0.8, 0.9],
    "Sequence_LowThredholdForOneStage" : [1.2, 1.2, 1.5, 1.5],
    "Sequence_HighThredholdForOneStage" : [1.0, 1.0, 1.2, 1.2],
    "Sequence_MATCH_RESULT": ["./GraphMatchkpl/GraphMatch1.pkl","./GraphMatchkpl/GraphMatch2.pkl","./GraphMatchkpl/GraphMatch3.pkl"],
    "MISS_MATCH_TARGETS": "./ResForTXTandVideo/MissMatchList.txt",
    "FRAMEINTERVAL": 5
}


#准备数据
def PreparetheData():

    FeatureRootPath='./demo_output/'
    OtherInfoRootPath='./labels/'
    # AllInfoPath='./demo/AllinfoPath.csv'

    UplevelPath=os.listdir(FeatureRootPath)
    # OtherInfoUplevel=os.listdir(OtherInfoRootPath)

    for EveryUplevelPath in UplevelPath:
        FeatPath=os.listdir(FeatureRootPath+EveryUplevelPath)
        TargetNumber=0


        
        for EveryFeatPath in FeatPath:
            with open(FeatureRootPath + EveryUplevelPath + '/' + EveryFeatPath, 'r') as f:
                reader = csv.reader(f)
                result = list(reader)
                # print(result[0])


            if((Nstr(EveryFeatPath)-Nstr(EveryUplevelPath)).split('.',1)[0]==''):
                TargetNumber=0
            else:
                TargetNumber=int((Nstr(EveryFeatPath)-Nstr(EveryUplevelPath)).split('.',1)[0])

            with open(OtherInfoRootPath  + EveryUplevelPath + '.txt', 'r') as f:
                OtherInfoIndex=0
                while OtherInfoIndex <TargetNumber-1:
                    line = f.readline()
                    OtherInfoIndex=OtherInfoIndex+1
                line = f.readline().replace("\n", "").split(' ')
                # print(line)
                # print(type(EveryUplevelPath))
                # print(type(TargetNumber))
                # print(type(line))
                # print(type(result))

                FinalResult=[]
                
                str1=(Nstr(EveryUplevelPath)-Nstr('Camera')).split('.',1)[0]
                CameraNumber=int(str1[0])
                str2=str1.split('frame',1)[1]
                FrameNumber=int(str2)

                FinalResult.append(CameraNumber)

                FinalResult.append(FrameNumber)

                if(TargetNumber==0):
                    FinalResult.append(int(1))
                else:
                    FinalResult.append(TargetNumber)

                line=list(map(float,line))
                for item in line:
                    FinalResult.append(item)
                
                result=list(map(float,result[0]))
                for item in result:
                    FinalResult.append(item)

                
                csvFile = open(config['AllInfoPath'], "a",newline='')
                writer = csv.writer(csvFile)
                writer.writerow(FinalResult)
                # print(FinalResult)


def ReadData():
    all_data_frames=[]
    CSVPath=config['AllInfoPath'] #文件路径
    AllInfoContent=open(CSVPath).readlines()
    TotalLenthofCrops = len(AllInfoContent)
    image = cv2.imread(config['SAMPLE_PATH'])
    size = image.shape
    w = size[1] 
    h = size[0]

    start_frame = AllInfoContent[0].split(",")[0]
    end_frame = AllInfoContent[-1].split(",")[0]

    for frame in range(start_frame, end_frame + 1):
        all_data_frames.append(
            {
                "frame": frame,
                "targets": []
            }
        )

    for data in range(TotalLenthofCrops):
        tempList=np.array(AllInfoContent[data].split(','))

        VisualFeature=[float(x) for x in tempList[9:]]      
                               #场景号               帧号            目标号              x                  y                         w                      h            置信度       视觉特征
        CurrentTarget=Target(int(tempList[0]),int(tempList[1]),int(tempList[2]),float(tempList[4])*w,float(tempList[5])*h,float(tempList[6])*w,float(tempList[7])*h,float(tempList[8]),VisualFeature)
        
        # if CurrentTarget.frame > 15:
        #     break
        if CurrentTarget.confidence < config["DETECTION_CONFIDENCE"]:
            continue

        all_data_frames[CurrentTarget.frame - 1]["targets"].append(CurrentTarget)
    
    return  all_data_frames


def PrepareMOTData():

    # 定义文件路径
    path_main = r'D:/PaperForReId/MainFlowforMHTandMWIS/MOT17-02-SDP/det'
    file1_path = f'{path_main}/MOT17-02-SDP.txt'
    file2_path = f'{path_main}/MOT17-02-SDP_ftrs.txt'
    output_csv_path = f'{path_main}/MOTdata.csv'

    # 读取文件内容
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    # 拼接对应行数据
    data = []
    for line1, line2 in zip(lines1, lines2):
        line1 = line1.strip()  # 去除行尾的换行符
        line2 = line2.strip()  # 去除行尾的换行符
        combined_line = f'{line1},{line2}'  # 拼接行数据
        data.append(combined_line)

    # 写入到 CSV 文件
    with open(output_csv_path, 'w') as csv_file:
        csv_file.write('MOT17-02-SDP,MOT17-02-SDP_ftrs\n')  # 写入 CSV 表头
        for line in data:
            csv_file.write(line + '\n')  # 写入每行数据

    # 打印完成提示
    # print(f'拼接数据已保存到 {output_csv_path}')


def ReadMOTData():
    all_data_frames=[]
    path_main = r'D:/PaperForReId/MainFlowforMHTandMWIS/MOT17-02-SDP/det'
    output_csv_path = f'{path_main}/MOTdata.csv'

    data = pd.read_csv(output_csv_path)
    start_frame = int(data.iloc[0, 0])
    end_frame = int(data.iloc[-1, 0])

    for frame in range(start_frame, end_frame + 1):
        all_data_frames.append(
            {
                "frame": frame,
                "targets": []
            }
        )

    with open(output_csv_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        count = 0

        for index, row in enumerate(csv_reader):
            float_row = [float(value) for value in row]

            count += 1
                                   #场景号       帧号                目标号                       x              y             w             h            置信度     视觉特征
            CurrentTarget=Target(   int('1'),   int(float_row[0]), int(count), float(float_row[2]),float(float_row[3]),float(float_row[4]),float(float_row[5]),float(float_row[6]),float_row[9:])
            

            # if CurrentTarget.frame > 600:
            #     break
            if CurrentTarget.confidence < config["DETECTION_CONFIDENCE"]:
                continue
            
            all_data_frames[CurrentTarget.frame - 1]["targets"].append(CurrentTarget)

    
    return  all_data_frames






# 准备数据
def PrepareGMOTdata():
  txt_to_csv(config["DATAINPUTPATH"],config["DATAOUTPUTPATH"])

def txt_to_csv(txt_folder_path, output_csv_path):

    txt_file_path = txt_folder_path
    df = pd.read_csv(txt_file_path, delimiter=',')  # 假设txt文件是以制表符分隔

# 保存为csv文件
    csv_file_path = output_csv_path
    df.to_csv(csv_file_path, index=False)

#格式化读取
def ReadGMOTdata():
    all_data_frames=[]
    CSVPath=config['DATAOUTPUTPATH'] #文件路径
    AllInfoContent=open(CSVPath).readlines()
    TotalLenthofCrops = len(AllInfoContent)
    # image = cv2.imread(config['SAMPLE_PATH'])
    # size = image.shape
    # w = size[1] 
    # h = size[0]

    start_frame = AllInfoContent[0].split(",")[0]
    end_frame = AllInfoContent[-1].split(",")[0]

    for frame in range(int(start_frame), int(end_frame) + 1):
        all_data_frames.append(
            {
                "frame": frame,
                "targets": []
            }
        )

    for data in range(TotalLenthofCrops):
        tempList=np.array(AllInfoContent[data].split(','))

                               #场景号               帧号            目标号              x                  y                     w              h              conf
        CurrentTarget=Target(int(1            ),int(tempList[0]),int(data),float(tempList[2]),float(tempList[3]),float(tempList[4]),float(tempList[5]),None)
        all_data_frames[CurrentTarget.frame]["targets"].append(CurrentTarget)
    
    return  all_data_frames
