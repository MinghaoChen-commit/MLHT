from tools.settings import *

def GetAllTarget(UnmatchedPool):
    AllTarget=[]
    AllTargetList = []
    for frame in UnmatchedPool:
        targets = frame["targets"]
        for target in targets:
            AllTarget.append(target)
            AllTargetList.append([target.id,target.x,target.y,target.width,target.height,target.frame])
    return AllTargetList


#批量处理
def read_all_files_in_directory(directory_path):
    # 获取目录下的所有文件路径
    all_files = glob.glob(os.path.join(directory_path, '*'))
    # print(all_files)
    return all_files
    # 读取文件内容
    # for file_path in all_files:
    #     if os.path.isfile(file_path):
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             content = file.read()
    #             print(f"Contents of {file_path}:")
    #             print(content)
    #             print("\n" + "="*80 + "\n")


def AllPathDet():
    #自制检测
    ListLabel = read_all_files_in_directory('D:\\PaperForReId\\Yolov8Api\\det')
    #GT
    ListLabel = read_all_files_in_directory('D:\\PaperForReId\\GMOT40\GMOT40\\track_label')
    ListPicture = read_all_files_in_directory(config["PICTUREROOTPATH"])
    ListPicture2 = []
    visualpath = []
    ListCSV = []
    txtRES = []
    for EveryPath in ListLabel:
        name = Path(EveryPath).stem
        EveryPath = 'D:\\PaperForReId\\GMOT40\\GMOT40\\GenericMOT_JPEG_Sequence\\' + name + "\\img1\\"
        ListPicture2.append(EveryPath)
        
      
        EveryPath2 = 'D:\\PaperForReId\\Yolov8Api\\tempcsv\\' + name  + ".csv"
        ListCSV.append(EveryPath2)

        EveryPath3 = 'D:\\PaperForReId\\Yolov8Api\\vis\\' + name  + "/"
        visualpath.append(EveryPath3)
        os.makedirs(EveryPath3, exist_ok=True)
        
        EveryPath4 = 'D:\\PaperForReId\\Yolov8Api\\vis\\' + name  + "/" + name + ".txt"
        txtRES.append(EveryPath4)


    return ListLabel, ListCSV, ListPicture2,visualpath,txtRES


#读所有路径
def AllPath():
    ListLabel = read_all_files_in_directory(config["LABELROOTPATH"])
    ListPicture = read_all_files_in_directory(config["PICTUREROOTPATH"])
    ListPicture2 = []
    visualpath = []
    ListCSV = []
    txtRES = []
    for EveryPath in ListPicture:
        name = Path(EveryPath).stem
        EveryPath += "\\img1\\"
        ListPicture2.append(EveryPath)
        
      
        EveryPath2 = config["CSVROOTPATH"] + name  + ".csv"
        ListCSV.append(EveryPath2)

        EveryPath3 = 'D:\\PaperForReId\\AllVisRes\\gt\\MHT_Graph_Relink\\' + name  + "/"
        visualpath.append(EveryPath3)
        os.makedirs(EveryPath3, exist_ok=True)
        
        EveryPath4 = 'D:\\PaperForReId\AllVisRes\\gt\\MHT_Graph_Relink' + "/" + name + ".txt"
        txtRES.append(EveryPath4)


    return ListLabel, ListCSV, ListPicture2,visualpath,txtRES

#准备GMOT数据
def PrepareGMOTdataWithPath(Label,csvname):
  txt_to_csv(Label,csvname)

#读取GMOT数据   
def ReadGMOTdataWith(csvname):
    all_data_frames=[]
    CSVPath=csvname #文件路径
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
        
        CurrentTarget.gtid = tempList[1]

        all_data_frames[CurrentTarget.frame]["targets"].append(CurrentTarget)
    
    return  all_data_frames

def draw_points(image_size, points, color, radius, save_path):
    # 创建一个指定大小的黑色图像
    # image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    # 在图像上画点
    for point in points:
        cv2.circle(image, (int(point[0]),int( point[1])), radius, color, -1)

    # 保存图像
    cv2.imwrite(save_path, image)


def CreateFakeData():
    # StatorPoint1 =[[50,50,5,5],[50,540,5,5],[50,1030,5,5],[1870,50,5,5],[1870,1030,5,5],[1870,540,5,5],[960,50,5,5],[960,1030,5,5]]
    # StatorPoint1 =[[50,50,5,5],[50,540,5,5],[50,1030,5,5],[1870,540,5,5],[960,50,5,5],[960,1030,5,5]]
    StatorPoint1 =[[1000,600,5,5],[1100,500,5,5],[900,500,5,5],[1050,400,5,5],[950,400,5,5],[1150,300,5,5]]
    all_data_frames=[]

    for frame in range(0, 100):
        all_data_frames.append(
            {
                "frame": frame,
                "targets": []
            }
        )
    
    for data in range(0,100):
        for index in range(len(StatorPoint1)):
                                           #场景号               帧号            目标号              x                  y                            w              h              conf
            CurrentTarget=Target(int(1            ),int(data ),int((len(StatorPoint1)+1)  * data + index),float(StatorPoint1[index][0]),float(StatorPoint1[index][1]),float(StatorPoint1[index][2]),float(StatorPoint1[index][3]),None)
            all_data_frames[CurrentTarget.frame]["targets"].append(CurrentTarget)
        
        CurrentTarget=Target(int(1            ),int(data ),int((len(StatorPoint1)+1)*(data + 1) -1 ),float(150 + 15 * data),float(1030),float(5),float(5),None)
        all_data_frames[CurrentTarget.frame]["targets"].append(CurrentTarget)
        # 1030+10*np.sin((data+0.5) * 3.14)


    for index,framedata in enumerate(all_data_frames):
        Points =[]
        for target in framedata["targets"]:
            Points.append((target.x, target.y))
        draw_points(image_size = (1920,1080),  color = (0,255,0), radius =10 , save_path = './FakeData/' + str(index).zfill(6) +'.jpg', points = Points)

    return  all_data_frames
    

#制造漏检误检数据
def Notall_data_GMOT(all_data_GMOT, JianGe, NumberOfLossTarget):
    Not_All_Data_GMOT = []
    Rest_Data_GMOT =[]
    for index,frame in enumerate(all_data_GMOT):
        if(index % JianGe == 0):
            random_numbers_list = random.sample(range(0, len(frame["targets"]) + 1), int(len(frame["targets"]) - NumberOfLossTarget))
            Oneframe={
                    "frame": frame["frame"],
                    "targets": []}
            RestOneframe={
                    "frame": frame["frame"],
                    "targets": []}
        
            for index, target in enumerate(frame["targets"]):
                if index in random_numbers_list:
                    Oneframe["targets"].append(target)
                else:
                    RestOneframe["targets"].append(target)

            Not_All_Data_GMOT.append(Oneframe)
            Rest_Data_GMOT.append(RestOneframe)
        else:
            Not_All_Data_GMOT.append(frame)
    
    return Not_All_Data_GMOT,Rest_Data_GMOT

def calculate_iou(bbox1, bbox2):
    x1_min, y1_min, w1, h1 = bbox1
    x2_min, y2_min, w2, h2 = bbox2
    
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    iou = inter_area / (area1 + area2 - inter_area)
    
    return iou

def merge_bboxes(bbox1, bbox2):
    x1_min, y1_min, w1, h1 = bbox1
    x2_min, y2_min, w2, h2 = bbox2
    
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    merged_x_min = min(x1_min, x2_min)
    merged_y_min = min(y1_min, y2_min)
    merged_x_max = max(x1_max, x2_max)
    merged_y_max = max(y1_max, y2_max)
    
    merged_w = merged_x_max - merged_x_min
    merged_h = merged_y_max - merged_y_min
    
    return [merged_x_min, merged_y_min, merged_w, merged_h]

def UnionIOU(all_data_GMOT,IOUThredhold):
    Union_data = []
    for frame in all_data_GMOT:
        framedata = frame["targets"]
        Oneframe={
                "frame": frame["frame"],
                "targets": []}

        for target1 in framedata:
            for target2 in framedata:
                if(target1.id != target2.id ):
                        IOU= calculate_iou(target1.bbox,target2.bbox)
                        if(IOU > IOUThredhold):
                            merged_bbox = merge_bboxes(target1.bbox, target2.bbox)
                                                    #场景号             帧号            目标号              x                  y                     w              h              conf
                            CurrentTarget=Target(int(1            ),int(frame["frame"]),int(target1.id),float(merged_bbox[0]),float(merged_bbox[1]),float(merged_bbox[2]),float(merged_bbox[3]),None)
                            Oneframe["targets"].append(CurrentTarget)
                            print(str(frame["frame"]) + " "+ str(target1.id) +" " + str(target2.id) )
                        else:
                             
                            if(target1 not in Oneframe["targets"]):
                                 Oneframe["targets"].append(target1)
                            if(target2 not in Oneframe["targets"]):
                                 Oneframe["targets"].append(target2)
        Union_data.append(Oneframe)

    return Union_data