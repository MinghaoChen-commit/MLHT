
from AllCompoent.MWIS import *
from AllCompoent.ShowResult import *
from AllCompoent.GraghMatch import *
from AllCompoent.interface import *
from AllCompoent.relink import *
from AllCompoent.Cluster import *

from tools.settings import *




        




#核心函数
def BuildTheFroestWithGraphMatchV4(all_data,Seqlist):
    trajectory_forest = []
    graphMatchList = []
    TotalUnMatchTargetsList = []
    # collision_set = []
    
    for data in all_data:
        frame = data["frame"]
        targets = data["targets"]

        trajectory_forest_new = deepcopy(trajectory_forest)
               
        graph_current = regenerateGraphwithFrame(targets)    
        graph_previous = regenerateGraphwithForest(trajectory_forest, graphMatchList, frame)

        graph_current, graph_previous = regenerateGraphWithThreshold(graph_current, graph_previous)

        graphMatchList_temp = []
        n1 = len(graph_current)
        n2 = len(graph_previous)
        if n1 * n2 > 0:
            start_time = time.time()
            
            for index, c in enumerate(config["Sequence_alpha"]):
                parameters = {}
                #region 图匹配参数配置
                parameters["alpha"] = config["Sequence_alpha"][index]
                parameters["dist_errorthredhold"] = config["Sequence_dist_errorthredhold"][index]
                parameters["iouthredhold"] = config["Sequence_iouthredhold"][index]
                parameters["SupportMaxrixthredhold"] = config["Sequence_SupportMaxrixthredhold"][index]
                parameters["Sequence_LowThredholdForOneStage"] = config["Sequence_LowThredholdForOneStage"][index]
                parameters["Sequence_HighThredholdForOneStage"] = config["Sequence_HighThredholdForOneStage"][index]
         

                graphMatch = SupportMatrixMatchV2(graph_previous, graph_current, parameters)
                # print(graphMatch)
                if(len(graphMatch) == 0):
                     print('-------------------------iou---------------------------------------------------')
                     print('-------------------------iou---------------------------------------------------')
                     print('-------------------------iou---------------------------------------------------')
                     print('-------------------------iou---------------------------------------------------')
                     print('-------------------------iou---------------------------------------------------')
                     print('-------------------------iou---------------------------------------------------')
                     print('-------------------------iou---------------------------------------------------')
                     
                # ProducePairRes(frame, frame-1, graphMatch, all_data, c, Seqlist)
                # ProducePairResV2(frame, frame-1, graphMatch, all_data, c, Seqlist)
                graphMatchList_temp += graphMatch


              
            graphMatchList_temp = list(set(graphMatchList_temp))


            Matched_Current_index = [y for x, y in graphMatchList_temp]
                
            All_current_index = [x[0] for x in graph_current]

            UnMatched_current_index = [x for x in All_current_index if x not in Matched_Current_index]


            UnMatchTargetsList = []
            for previous_id in UnMatched_current_index:
                    UnmatchedCurrentTarget = FindMatchTargets(all_data,previous_id,frame)
                    UnMatchTargetsList.append(UnmatchedCurrentTarget)
            
            if(UnMatchTargetsList != []):
                UnMatchTargetsData = {}
                UnMatchTargetsData["frame"] = frame
                UnMatchTargetsData["targets"] = UnMatchTargetsList
                TotalUnMatchTargetsList.append(UnMatchTargetsData)



            graphMatchList += graphMatchList_temp


            end_time = time.time()
            print('【Graph Match】', n1, n2, end_time-start_time)
            
        for branch in trajectory_forest:
            target = branch.targets[-1]
            graphMatchListFiltered = list(filter(lambda match: match[0] == target.id, graphMatchList_temp))
            for match in graphMatchListFiltered:
                target_append = list(filter(lambda target: match[1] == target.id, targets))
                if(target_append[0].gtid != target.gtid):
                     continue     
                branch_new = deepcopy(branch)
                if len(target_append) > 0:
                    branch_new.targets.append(target_append[0])
                    branch_new.branch_id = GetBranchID()
                    trajectory_forest_new.append(branch_new)

        #新建树
        for target in targets:
            trajectory_branch = Trajectory(target, GetBranchID())
            trajectory_forest_new.append(trajectory_branch)



  

        VisID(trajectory_forest, frame)
        # print("------------------------------collision_set-----------------------------")
        # VisSet(collision_set)
                
        if(len(TotalUnMatchTargetsList) != 0):
            if(TotalUnMatchTargetsList[-1]["frame"] - TotalUnMatchTargetsList[0]["frame"] >= 5):
                SelectedData = GetAllTarget(TotalUnMatchTargetsList)
                result = DBSCAN_Anaylse(SelectedData)
                TotalUnMatchFroest = []

                BuildTreeFromDBSCAN(result,TotalUnMatchTargetsList, TotalUnMatchFroest)
    

                trajectory_forest_temp =  deepcopy(trajectory_forest_new)
                
                # for everybranch in TotalUnMatchFroest:
                #     trajectory_forest_temp.append(everybranch)
        


                RELinkFroest1,ClusterFroset2 = ReLinkByAFLinkWithlessCalculate4(trajectory_forest_temp,TotalUnMatchFroest,frame)
                # VisID(RELinkFroest1, frame)

                RELinkFroest1List = []
                for branch in ClusterFroset2:
                    for target in branch.targets:
                        RELinkFroest1List.append(target.sid)
                
                TotalUnMatchTargetsListList2 = []
                for frames in TotalUnMatchTargetsList:
                    UnMatchTargetsData = {}
                    UnMatchTargetsData["frame"] = frames['frame']
                    UnMatchTargetsData["targets"] = []
                    for  Aframedate in frames['targets']:
                        if Aframedate.sid not in RELinkFroest1List:
                                UnMatchTargetsData["targets"].append(Aframedate)
                    
                    TotalUnMatchTargetsListList2.append(UnMatchTargetsData)


                TotalUnMatchTargetsList = list(filter(lambda branch: frame - branch['frame'] <= 5, TotalUnMatchTargetsListList2))

                
          

        
        trajectory_forest = deepcopy(trajectory_forest_new)
        
        if len(trajectory_forest) > config["BRANCH_COUNT_THRESHOLD"]:   
            start_time = time.time()
            print("------------------------------BEFORE-----------------------------")
            VisID(trajectory_forest, frame ,frame)
            collision_set = GenerateCollision(trajectory_forest)
            trajectory_forest = max_weight_independent_set_lp(trajectory_forest, collision_set, frame)
            collision_set = []
            end_time = time.time()
            print('【MWIS】', end_time-start_time)
            VisID(trajectory_forest, frame ,frame)
            print("-----------------------------------------------------------")
        
        

    
    collision_set = GenerateCollision(trajectory_forest)
    trajectory_forest = max_weight_independent_set_lp(trajectory_forest, collision_set, frame)

    trajectory_forest_temp2 = []
    for branch in trajectory_forest:
                if(len(branch.targets) < int(frame) + 1):
                    trajectory_forest_temp2.append(branch)
                
    RELinkFroest2 , _ = ReLinkByAFLinkWithlessCalculate4(trajectory_forest_temp2,trajectory_forest_temp2,frame)
    trajectory_forest.extend(RELinkFroest2)
    VisID(trajectory_forest, frame,frame)
    collision_set = GenerateCollision(trajectory_forest)
    trajectory_forest = max_weight_independent_set_lp(trajectory_forest, collision_set, frame)
    collision_set = []
    VisID(trajectory_forest,frame, frame)


    
    return trajectory_forest




def GraphMatchCompoent(graph_current,graph_previous,all_data,frame,seqlist):
        graphMatchList_temp = []
        UnMatchTargetsList = []
        n1 = len(graph_current)
        n2 = len(graph_previous)
        if n1 * n2 > 0:
            start_time = time.time()
            
            for index, c in enumerate(config["Sequence_alpha"]):
                parameters = {}
                #region 图匹配参数配置
                parameters["alpha"] = config["Sequence_alpha"][index]
                parameters["dist_errorthredhold"] = config["Sequence_dist_errorthredhold"][index]
                parameters["iouthredhold"] = config["Sequence_iouthredhold"][index]
                parameters["SupportMaxrixthredhold"] = config["Sequence_SupportMaxrixthredhold"][index]
                parameters["Sequence_LowThredholdForOneStage"] = config["Sequence_LowThredholdForOneStage"][index]
                parameters["Sequence_HighThredholdForOneStage"] = config["Sequence_HighThredholdForOneStage"][index]
         

                graphMatch = SupportMatrixMatchV2(graph_previous, graph_current, parameters)
                print(graphMatch)
                
                graphMatchList_temp += graphMatch


              
            graphMatchList_temp = list(set(graphMatchList_temp))


            Matched_Current_index = [y for x, y in graphMatchList_temp]
                
            All_current_index = [x[0] for x in graph_current]

            UnMatched_current_index = [x for x in All_current_index if x not in Matched_Current_index]
  
            for previous_id in UnMatched_current_index:
                    UnmatchedCurrentTarget = FindMatchTargets(all_data,previous_id,frame)
                    UnMatchTargetsList.append(UnmatchedCurrentTarget)
           
            end_time = time.time()
            print('【Graph Match】', n1, n2, end_time-start_time)
        
        return  graphMatchList_temp , UnMatchTargetsList
     

def GraphMatchCompoentV2(graph_current,graph_previous,all_data,frame,Seqlist):
        graphMatchList_temp = []
        UnMatchTargetsList = []
        n1 = len(graph_current)
        n2 = len(graph_previous)
        if n1 * n2 > 0:
            start_time = time.time()
            graphMatchList = []
            if(len(graph_current) > 3 and len(graph_previous) > 3):
                for index, c in enumerate(config["Sequence_alpha"]):
                    parameters = {}
                    #region 图匹配参数配置
                    parameters["alpha"] = config["Sequence_alpha"][index]
                    parameters["dist_errorthredhold"] = config["Sequence_dist_errorthredhold"][index]
                    parameters["iouthredhold"] = config["Sequence_iouthredhold"][index]
                    parameters["SupportMaxrixthredhold"] = config["Sequence_SupportMaxrixthredhold"][index]
                    parameters["Sequence_LowThredholdForOneStage"] = config["Sequence_LowThredholdForOneStage"][index]
                    parameters["Sequence_HighThredholdForOneStage"] = config["Sequence_HighThredholdForOneStage"][index]
                    graphMatch = SupportMatrixMatchV2(graph_previous, graph_current, parameters)
                    graphMatchList.append(graphMatch)
                    sorted_list = sorted(graphMatchList[index], key=lambda x: x[0], reverse=True)

                


            if(len(graphMatchList) != 0):  
                graphMatchList_temp = list(set(graphMatchList[0])&set(graphMatchList[1])&set(graphMatchList[2]))
                sorted_list2 = sorted(graphMatchList_temp, key=lambda x: x[0], reverse=True)
            # print(sorted_list2)

            Matched_Current_index = [y for x, y in graphMatchList_temp]
                
            All_current_index = [x[0] for x in graph_current]

            UnMatched_current_index = [x for x in All_current_index if x not in Matched_Current_index]
  
            for previous_id in UnMatched_current_index:
                    UnmatchedCurrentTarget = FindMatchTargets(all_data,previous_id,frame)
                    UnMatchTargetsList.append(UnmatchedCurrentTarget)
           
            end_time = time.time()
            print('【Graph Match】', n1, n2, end_time-start_time)
        
        return  graphMatchList_temp , UnMatchTargetsList



def ClusterCompoent(TotalUnMatchTargetsList,trajectory_forest_new,frame):

                SelectedData = GetAllTarget(TotalUnMatchTargetsList)
                result = DBSCAN_Anaylse(SelectedData)
                TotalUnMatchFroest = []

                BuildTreeFromDBSCAN(result,TotalUnMatchTargetsList, TotalUnMatchFroest)
    

        
                trajectory_forest_temp =  deepcopy(trajectory_forest_new)
                
                for everybranch in TotalUnMatchFroest:
                    trajectory_forest_temp.append(everybranch)
        
               

                RELinkFroest1,ClusterFroset2 = ReLinkByAFLinkWithlessCalculate2(trajectory_forest_temp,TotalUnMatchFroest,frame)
              

                RELinkFroest1List = []
                for branch in ClusterFroset2:
                    for target in branch.targets:
                        RELinkFroest1List.append(target.sid)
                
                TotalUnMatchTargetsListList2 = []
                for frames in TotalUnMatchTargetsList:
                    UnMatchTargetsData = {}
                    UnMatchTargetsData["frame"] = frames['frame']
                    UnMatchTargetsData["targets"] = []
                    for  Aframedate in frames['targets']:
                        if Aframedate.sid not in RELinkFroest1List:
                                UnMatchTargetsData["targets"].append(Aframedate)
                    
                    TotalUnMatchTargetsListList2.append(UnMatchTargetsData)


                TotalUnMatchTargetsList = list(filter(lambda branch: frame - branch['frame'] <= 5, TotalUnMatchTargetsListList2))

                
                trajectory_forest_new.extend(RELinkFroest1)

        
                return  trajectory_forest_new, TotalUnMatchTargetsList
     

def MWISCompoent(trajectory_forest,frame):
        start_time = time.time()


        collision_set = GenerateCollision(trajectory_forest)
        trajectory_forest = max_weight_independent_set_lp(trajectory_forest, collision_set, frame)
        collision_set = []

        for i in range(0,3):
            trajectory_forest_temp2 = []
            for branch in trajectory_forest:
                        if(len(branch.targets) < int(frame) + 1):
                            trajectory_forest_temp2.append(branch)
                    
            RELinkFroest2 ,_  = ReLinkByAFLinkWithlessCalculate3(trajectory_forest_temp2,trajectory_forest_temp2,frame)

            trajectory_forest.extend(RELinkFroest2)
            # VisID(trajectory_forest, frame)
            collision_set = GenerateCollision(trajectory_forest)
            trajectory_forest = max_weight_independent_set_lp(trajectory_forest, collision_set, frame)
            collision_set = []
            # VisID(trajectory_forest, frame)
        
        return trajectory_forest


def MWISCompoentV2(trajectory_forest,frame):
        start_time = time.time()
        print("------------------------------BEFORE-----------------------------")
        VisID(trajectory_forest, frame ,frame)

        collision_set = GenerateCollision(trajectory_forest)
        trajectory_forest = max_weight_independent_set_lp(trajectory_forest, collision_set, frame)
        collision_set = []
        end_time = time.time()
        print('【MWIS】', end_time-start_time)
        VisID(trajectory_forest, frame ,frame)
        print("-----------------------------------------------------------")



        # MHTReduction(trajectory_forest, frame)

        for i in range(0,1):
            trajectory_forest_temp2 = []
            for branch in trajectory_forest:
                        if(len(branch.targets) < int(frame) + 1):
                            trajectory_forest_temp2.append(branch)
                    
            RELinkFroest2 ,_  = ReLinkByAFLinkWithlessCalculate4(trajectory_forest_temp2,trajectory_forest_temp2,frame)

            trajectory_forest.extend(RELinkFroest2)
            # VisID(trajectory_forest, frame)
            collision_set = GenerateCollision(trajectory_forest)
            trajectory_forest = max_weight_independent_set_lp(trajectory_forest, collision_set, frame)
            collision_set = []
            # VisID(trajectory_forest, frame)
        
        return trajectory_forest

     


def BuildTheFroestWithGraphMatchV5(all_data,Seqlist):
    trajectory_forest = []
    graphMatchList = []
    TotalUnMatchTargetsList = []

    
    for data in all_data:
        frame = data["frame"]
        targets = data["targets"]
        if(len(targets) != 0):
            trajectory_forest_new = deepcopy(trajectory_forest)
            graph_current = regenerateGraphwithFrame(targets)    
            graph_previous = regenerateGraphwithForest(trajectory_forest, graphMatchList, frame)
            graph_current, graph_previous = regenerateGraphWithThreshold(graph_current, graph_previous)
      
            graphMatchList_temp , UnMatchTargetsList = GraphMatchCompoentV2(graph_current,graph_previous,all_data,frame,Seqlist)

            if(UnMatchTargetsList != []):
                UnMatchTargetsData = {}
                UnMatchTargetsData["frame"] = frame
                UnMatchTargetsData["targets"] = UnMatchTargetsList
                TotalUnMatchTargetsList.append(UnMatchTargetsData)

            graphMatchList += graphMatchList_temp

            for branch in trajectory_forest:
                target = branch.targets[-1]
                graphMatchListFiltered = list(filter(lambda match: match[0] == target.id, graphMatchList_temp))
                for match in graphMatchListFiltered:
                    target_append = list(filter(lambda target: match[1] == target.id, targets))   
                    if(len(target_append) != 0):
                        # if target_append[0].gtid == target.gtid:
                            branch_new = deepcopy(branch)
                            if len(target_append) > 0:
                                branch_new.targets.append(target_append[0])
                                branch_new.branch_id = GetBranchID()
                                trajectory_forest_new.append(branch_new)
                            
            
            #新建树
            for target in targets:
                trajectory_branch = Trajectory(target, GetBranchID())
                trajectory_forest_new.append(trajectory_branch)
            
            
   
            
            trajectory_forest =deepcopy(trajectory_forest_new)
            


            if len(trajectory_forest) > config["BRANCH_COUNT_THRESHOLD"]: 
                trajectory_forest = MWISCompoentV2(trajectory_forest,frame)
                VisID(trajectory_forest, frame ,frame)

        
    


    

    trajectory_forest=  MWISCompoentV2(trajectory_forest,frame)
    VisID(trajectory_forest, frame)



    
    return trajectory_forest



def MHRelinkV3(CostList, froest1, froest2,thr):
    result = []
    ClusterFroset2 = []
    for cost in CostList:
        if cost[2] > thr:
            Branch_id_i = froest1[cost[0]].targets
            Branch_id_j = froest2[cost[1]].targets

            # if Branch_id_i[-1].gtid != Branch_id_j[-1].gtid :
            #      continue
            
            # 合并两个有序列表
            IntergratedBranch = []
            curi, curj = 0, 0
            len_i, len_j = len(Branch_id_i), len(Branch_id_j)
            
            while curi < len_i and curj < len_j:
                if Branch_id_i[curi].frame < Branch_id_j[curj].frame:
                    IntergratedBranch.append(Branch_id_i[curi])
                    curi += 1
                else:
                    IntergratedBranch.append(Branch_id_j[curj])
                    curj += 1

            # 直接添加剩余部分
            if curi < len_i:
                IntergratedBranch.extend(Branch_id_i[curi:])
            if curj < len_j:
                IntergratedBranch.extend(Branch_id_j[curj:])
            
            trajectory_branch = Trajectory(IntergratedBranch, GetBranchID())
            result.append(trajectory_branch)

            ClusterFroset2.append(froest2[cost[1]])

    return result,ClusterFroset2



def ReLinkByAFLinkWithlessCalculate4(MHTaddCluster, Cluster, frame):
    HalfcostList = [] 
    costList =  LinkAWithB(MHTaddCluster,Cluster)

    for ConncetPair1 in costList:
            if ConncetPair1[1] >  ConncetPair1[0]:
                HalfcostList.append(ConncetPair1)
    
    sorted_HalfcostList = sorted(HalfcostList, key=lambda x: x[2], reverse=True)

    if frame <=3:
        sorted_HalfcostList=sorted_HalfcostList[:int(0.5*len(sorted_HalfcostList))]
                 
        forest,ClusterFroset2 = MHRelinkV3(sorted_HalfcostList,MHTaddCluster,Cluster,thr = 50)
    
    else:    
                
        forest,ClusterFroset2 = MHRelinkV3(sorted_HalfcostList,MHTaddCluster,Cluster,thr = 50)

        VisID(forest, frame)

    return forest,ClusterFroset2
                                     

 



def visualizeGMOTWithPath(final_trajectory, all_data_DJI, ListPicture2, VISUALPATH):
    FrameOriginzation={}
    for frame in range(int(all_data_DJI[0]["frame"]),int(all_data_DJI[-1]["frame"]+1)):
        FrameOriginzation[frame] = []
    
    for branch in final_trajectory:
        for target in branch.targets:
            FrameOriginzation[target.frame].append(VisualizeStructure(target,branch))
    
    for index, frame in enumerate(FrameOriginzation):
        if(FrameOriginzation[frame] != None):
            PicturePath=ListPicture2 + str(frame).zfill(6) + ".jpg"
            img = cv2.imread(PicturePath)
            for EveryBox in FrameOriginzation[frame]:
                x_lu = int(EveryBox.x - 0 * EveryBox.w)
                y_lu = int(EveryBox.y - 0 * EveryBox.h)
                x_rl = int(EveryBox.x + 1.0 * EveryBox.w)
                y_rl = int(EveryBox.y + 1.0 * EveryBox.h)

                # 绘制边界框和文本
                cv2.rectangle(img, (x_lu, y_lu), (x_rl, y_rl), EveryBox.color_for_bbox, thickness=config['BOUNDINGBOX_THICKNESS'])
                cv2.putText(img, str(EveryBox.id), (x_lu, y_lu + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, EveryBox.color_for_bbox, config['BOUNDINGBOX_THICKNESS'] - 1)

            OutputPicturePath=VISUALPATH + str(frame).zfill(6) + ".jpg"
            cv2.imwrite(OutputPicturePath, img)
            # print("success")

#保存eval结果
def SavetheResultWithPath(final_trajectory, RESpath):
    Evaluation_list=[]
    for index, trajectory in enumerate(final_trajectory):
        for target in trajectory.targets:     
            # 保存Results
            Evaluation_list.append([target.frame, index + 1, target.x, target.y, target.width, target.height, 1, -1, -1, -1])
    
    sorted_lists = sorted(Evaluation_list, key=lambda x: x[0])

    with open(RESpath, 'w') as f:
        for sublist in sorted_lists:
            line = ','.join(map(str, sublist))
            f.write(line + '\n')

    return True



def visualizeGMOTWithPath2(final_trajectory, all_data_DJI, ListPicture2, VISUALPATH):
    FrameOriginzation={}
    for frame in range(int(all_data_DJI[0]["frame"]),int(all_data_DJI[-1]["frame"]+1)):
        FrameOriginzation[frame] = []
    
    for branch in final_trajectory:
        for target in branch.targets:
            FrameOriginzation[target.frame].append(VisualizeStructure(target,branch))
    
    for index, frame in enumerate(FrameOriginzation):
        if(FrameOriginzation[frame] != None):
            PicturePath=ListPicture2 + str(frame).zfill(7) + ".jpg"
            img = cv2.imread(PicturePath)
            for EveryBox in FrameOriginzation[frame]:
                
                x_lu = int(EveryBox.x - 0.5 * EveryBox.w)
                y_lu = int(EveryBox.y - 0.5 * EveryBox.h)
                cv2.circle(img, (x_lu, y_lu), 2, EveryBox.color_for_bbox, thickness=config['BOUNDINGBOX_THICKNESS'])
              
            OutputPicturePath=VISUALPATH + str(frame).zfill(6) + ".jpg"
            cv2.imwrite(OutputPicturePath, img)
            # print("success")

#保存eval结果
def SavetheResultWithPath(final_trajectory, RESpath):
    Evaluation_list=[]
    for index, trajectory in enumerate(final_trajectory):
        for target in trajectory.targets:     
            # 保存Results
            Evaluation_list.append([target.frame, index + 1, target.x, target.y, target.width, target.height, 1, -1, -1, -1])
    
    sorted_lists = sorted(Evaluation_list, key=lambda x: x[0])

    with open(RESpath, 'w') as f:
        for sublist in sorted_lists:
            line = ','.join(map(str, sublist))
            f.write(line + '\n')

    return True


#MHT
def MHT(all_data, ListPicture2, VISUALPATH, RESpath, seqList):
    # 
    # 
    trajectory_forest = BuildTheFroestWithGraphMatchV5(all_data,seqList)

    visualizeGMOTWithPath(trajectory_forest, all_data, ListPicture2, VISUALPATH)
    SavetheResultWithPath(final_trajectory, RESpath)

    return(final_trajectory)


#MHT多进程主函数
def MHT_P(ListLabel, ListCSV, ListPicture2, VISUALlist, txtRES, i, seqList):
    PrepareGMOTdataWithPath(ListLabel[i], ListCSV[i])

    #MHT算法
    all_data_GMOT = ReadGMOTdataWith(ListCSV[i])

    
    final_trajectory = MHT(all_data_GMOT, ListPicture2[i], VISUALlist[i], txtRES[i],seqList)

    #生成视频
    # VideoBuilder()


def txt_to_csv(txt_folder_path, output_csv_path):

    txt_file_path = txt_folder_path
    df = pd.read_csv(txt_file_path, delimiter=',')  # 假设txt文件是以制表符分隔

# 保存为csv文件
    csv_file_path = output_csv_path
    df.to_csv(csv_file_path, index=False)


def PrepareGMOTdataWithPath(Label,csvname):
  txt_to_csv(Label,csvname)

#格式化读取

def list_files_without_extension(directory):
    try:
        with os.scandir(directory) as entries:
            files = [os.path.splitext(entry.name)[0] for entry in entries if entry.is_file()]
        return files
    except FileNotFoundError:
        print(f"指定的路径 {directory} 不存在")
        return []

def ReadGMOTdataWithSate(csvname,PicPath):
    all_data_frames=[]
    CSVPath=csvname #文件路径
    AllInfoContent=open(CSVPath).readlines()
    TotalLenthofCrops = len(AllInfoContent)
    # image = cv2.imread(config['SAMPLE_PATH'])
    # size = image.shape
    # w = size[1] 
    # h = size[0]
# 调用函数，指定路径
    files = list_files_without_extension(PicPath)
    
    start_frame = AllInfoContent[0].split(",")[0]
    end_frame = AllInfoContent[-1].split(",")[0]
    
    # start_frame = int(files[0])
    # end_frame = int(files[-1])

    for frame in range(int(start_frame), int(end_frame) + 1):
        all_data_frames.append(
            {
                "frame": frame,
                "targets": []
            }
        )

    for index,data in enumerate(range(TotalLenthofCrops)):
        tempList=np.array(AllInfoContent[data].split(','))

                               #场景号               帧号            目标号              x                  y                     w              h              conf
        CurrentTarget=Target(int(1            ),int(tempList[0]),int(index),float(tempList[2]),float(tempList[3]),float(tempList[4]),float(tempList[5]),float(tempList[6]))
        
        CurrentTarget.gtid = int(float(tempList[1]))
        

        all_data_frames[CurrentTarget.frame ]["targets"].append(CurrentTarget)

    
    return  all_data_frames


def AllPathDetVis():
    ListLabel = read_all_files_in_directory('')

    ListLabel = read_all_files_in_directory('')

    ListPicture2 = []
    visualpath = []
    ListCSV = []
    txtRES = []
    for EveryPath in ListLabel:
        name = Path(EveryPath).stem
        EveryPath = ''
        ListPicture2.append(EveryPath)
        
      
        EveryPath2 = ''
        ListCSV.append(EveryPath2)

        EveryPath3 = ''
        visualpath.append(EveryPath3)
        os.makedirs(EveryPath3, exist_ok=True)
        
        EveryPath4 = ''
        txtRES.append(EveryPath4)


    return ListLabel, ListCSV, ListPicture2,visualpath,txtRES



def list_directories(directory):
    List = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            print(os.path.basename(dir_name))  # 只打印文件夹名称，不包括路径
            List.append(dir_name)
    return List


#主函数
if __name__ == '__main__':

    AllList =list_directories('D')

    for i in range(39,40):
        ListLabel, ListCSV, ListPicture2, VISUALlist, txtRES = AllPathDetVis()
        PicPath = ''
        # PrepareGMOTdataWithPath(ListLabel[i], ListCSV[i])
        
        all_data_GMOT = ReadGMOTdataWithSate(ListCSV[i],PicPath)

        final_trajectory = MHT(all_data_GMOT,PicPath, VISUALlist[i], txtRES[i],AllList[i])

