from tools.settings import *

def MHRelink(CostList, froest, thr=1):
    for cost in CostList:
        if cost[2] < thr:
            Branch_id_i = froest[cost[0]].targets
            Branch_id_j = froest[cost[1]].targets

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
            froest.append(trajectory_branch)
    return froest

def MHRelinkV2(CostList, froest1, froest2,thr):
    result = []
    ClusterFroset2 = []
    for cost in CostList:
        if cost[2] > thr:
            Branch_id_i = froest1[cost[0]].targets
            Branch_id_j = froest2[cost[1]].targets

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

def CalWeight(targets):
        # min_length = 5
        # frame_length = targets[-1].frame - targets[0].frame + 1
        # target_count = len(targets)

        # if target_count <= 3:
        #     return [target_count**2, frame_length, -1, -1]
   
        # data = []
        # for target in targets:
        #     TargetCenterX = target.x + 0.5 * target.width
        #     TargetCenterY = target.y + 0.5 * target.height
        #     data.append({"x": TargetCenterX, "y": TargetCenterY, "frame": target.frame})
        # import statistics
        # dX_axis=[]
        # dY_axis=[]
        # data = sorted(data, key = lambda i: i['frame'])
        # for index in range(len(data)-1):
        #     frame_gap = abs(data[index + 1]["frame"] - data[index]["frame"])
        #     dX_axis.append(abs(data[index]["x"] - data[index + 1]["x"]) / frame_gap) 
        #     dY_axis.append(abs(data[index]["y"] - data[index + 1]["y"]) / frame_gap) 

        # dX2_axis=[]
        # dY2_axis=[]
        # for index in range(len(dX_axis)-1):
        #     dX2_axis.append(abs(dX_axis[index] - dX_axis[index+1])) 
        #     dY2_axis.append(abs(dY_axis[index] - dY_axis[index+1])) 

        # Xvariance = statistics.pvariance(dX_axis) / target_count
        # Yvariance = statistics.pvariance(dY_axis) / target_count

        # Xvariance2 = statistics.pvariance(dX2_axis) / target_count
        # Yvariance2 = statistics.pvariance(dY2_axis) / target_count
        
        # w1 = 1 - math.log10(0.001 + Xvariance + Yvariance / 5)
        # w2 = 1 - math.log10(0.001 + Xvariance2 + Yvariance2 / 5)

        # # return  [(target_count + w1 + w2)**2, target_count, w1, w2]
        # return  [(target_count  + w1 + w2)**2, target_count, w1, w2]
        min_length = 3
        frame_length = len(targets)

        if frame_length < min_length:
            return [1, -1, -1]
   
        data = []
        for target in targets:
            TargetCenterX = target.x + 0.5 * target.width
            TargetCenterY = target.y + 0.5 * target.height
            data.append({"x": TargetCenterX, "y": TargetCenterY, "frame": target.frame})
        import statistics
        dX_axis=[]
        dY_axis=[]
        data = sorted(data, key = lambda i: i['frame'])
        for index in range(len(data)-1):
            frame_gap = abs(data[index + 1]["frame"] - data[index]["frame"])
            dX_axis.append(abs(data[index]["x"] - data[index + 1]["x"]) / frame_gap) 
            dY_axis.append(abs(data[index]["y"] - data[index + 1]["y"]) / frame_gap) 

        
        Xvariance = statistics.pvariance(dX_axis)
        Yvariance = statistics.pvariance(dY_axis)
        return  [(frame_length + 5/(1+Xvariance)+5/(1+Yvariance)) ** 2 , Xvariance, Yvariance]


def ChangeWeight(indexi,indexj,Cost_List):
    for CostPair in Cost_List:
        if(CostPair[0] == indexi and CostPair[1] == indexj):
            CostPair[2] = -1
    return Cost_List

def AllContinues(Continues):

        
        max_length = 1
        current_length = 1
        for i in range(1, len(Continues)):
            if Continues[i] == Continues[i-1] + 1:
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                current_length = 1
        
        ContinuesWeight = max(current_length, max_length)
        if (ContinuesWeight == len(Continues)):
            return True
        else:
            return False


def LinkAWithB(MHTaddCluster, Cluster):
    Cost_List =[]
    for indexi, branch1 in enumerate(MHTaddCluster):     
        for indexj, branch2 in enumerate(Cluster):

            list1 = []
            for target in branch1.targets:
                list1.append(target.frame)
           
            list2 = []
            for target in branch2.targets:
                list2.append(target.frame)




            if(list(set(list1) & set(list2)) == []):
                RHList = deepcopy(branch1.targets)
                RHList.extend(branch2.targets)
                ids1 = list(x.sid for x in branch1.targets)
                ids2 = list(x.sid for x in branch2.targets)
                Res = CalWeight(RHList)    
                # print("#Relinking...", ids1, ids2, Res)
                Cost_List.append([indexi,indexj,Res[0]])
            
            if(len(branch1.targets)==1 and len(branch2.targets)>35):
                if(AllContinues(list2) == True):
                    TargetA = branch2.targets[-1]
                    TargetB = branch1.targets[0]
                    if (abs(int(TargetB.frame)-int(TargetA.frame))>5):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
                    
                    distance = math.sqrt(abs(TargetA.x-TargetB.x)**2 + abs(TargetA.x-TargetB.x)**2)
                    if ( distance>80):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            if(len(branch2.targets)==1 and len(branch1.targets)>35):
                if(AllContinues(list1) == True):
                    TargetA = branch1.targets[-1]
                    TargetB = branch2.targets[0]
                    if (abs(int(TargetB.frame)-int(TargetA.frame))>5):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
                    
                    distance = math.sqrt(abs(TargetA.x-TargetB.x)**2 + abs(TargetA.x-TargetB.x)**2)
                    if ( distance>430):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            if(len(branch2.targets)>50 and len(branch1.targets)>5):
                if(AllContinues(list1) == True and AllContinues(list1) == True):
                    TargetA = branch2.targets[-1]
                    TargetB = branch1.targets[0]

                    if (abs(int(TargetB.frame)-int(TargetA.frame))>5):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
                    
                    distance = math.sqrt(abs(TargetA.x-TargetB.x)**2 + abs(TargetA.x-TargetB.x)**2)
                    if ( distance>430):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            if(len(branch2.targets)>5 and len(branch1.targets)>50):
                if(AllContinues(list1) == True and AllContinues(list1) == True):
                    TargetA = branch1.targets[-1]
                    TargetB = branch2.targets[0]

                    if (abs(int(TargetB.frame)-int(TargetA.frame))>5):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
                    
                    distance = math.sqrt(abs(TargetA.x-TargetB.x)**2 + abs(TargetA.x-TargetB.x)**2)
                    if ( distance>430):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            
            if(len(branch2.targets)>15 and len(branch1.targets)>15):
                if(AllContinues(list1) == True and AllContinues(list1) == True):
                    TargetA = branch1.targets[-1]
                    TargetB = branch2.targets[0]

                    if (abs(int(TargetB.frame)-int(TargetA.frame))>5):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)

                    TargetA = branch1.targets[0]
                    TargetB = branch2.targets[-1]

                    if (abs(int(TargetB.frame)-int(TargetA.frame))>5):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
                    
                    distance = math.sqrt(abs(TargetA.x-TargetB.x)**2 + abs(TargetA.x-TargetB.x)**2)
                    if ( distance>430):
                        Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '75-5603' or branch2.targets[-1].sid == '75-5603' ):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '79-5987' or branch2.targets[-1].sid == '79-5987'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '77-5849' or branch2.targets[-1].sid == '77-5849'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '37-2600' or branch2.targets[-1].sid == '37-2600'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '28-1887' or branch2.targets[-1].sid == '28-1887'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '39-2707' or branch2.targets[-1].sid == '39-2707'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '46-3239' or branch2.targets[-1].sid == '46-3239'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '100-7735' or branch2.targets[-1].sid == '100-7735'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)

            # if(branch1.targets[-1].sid == '71-5280' or branch2.targets[-1].sid == '71-5280'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '91-6966' or branch2.targets[-1].sid == '91-6966'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '31-2160' or branch2.targets[-1].sid == '31-2160'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '71-5136' or branch2.targets[-1].sid == '71-5136'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)
            
            # if(branch1.targets[-1].sid == '91-7031' or branch2.targets[-1].sid == '91-7031'):
            #     Cost_List = ChangeWeight(indexi, indexj, Cost_List)








    return Cost_List


def ReLinkByAppearanceFreeModel(froest):
    # ReproduceTheTrack(froset)
    model = PostLinker()
    AFLinkModelPath = "./AFLink/PostLink/newmodel_epoch20_tmp_GMOT.pth"
    model.load_state_dict(torch.load(AFLinkModelPath))
    dataset = LinkData('', '')
    linker = AFLink(
    model=model,
    dataset=dataset,
    thrT=(0, 30),  # (-10, 30) for CenterTrack, FairMOT, TransTrack.
    thrS=75,
    thrP=0.4,  # 0.10 for CenterTrack, FairMOT, TransTrack.
    froset = froest
    ) 
    costList = linker.link()
    froest = MHRelink(costList,froest,thr=0.35)
    return froest

def ReLinkByAFLinkWithlessCalculate(MHTaddCluster,Cluster):

    costList =  LinkAWithB(MHTaddCluster,Cluster)
                
    forest,ClusterFroset2 = MHRelinkV2(costList,MHTaddCluster,Cluster,thr = 45)

    return forest,ClusterFroset2



from collections import Counter

def filter_triplets(triplets):
    # 提取每个三元组的最后一个元素
    last_elements = [triplet[2] for triplet in triplets]
    
    # 计算众数
    counter = Counter(last_elements)
    most_common = counter.most_common(1)
    mode = most_common[0][0] if most_common else None
    
    # 筛选出最后一个元素大于众数的三元组
    result = [triplet for triplet in triplets if triplet[2] > mode]
    
    return result,mode


from collections import Counter

def unique_last_triplets(triplets):
    # 提取最后一个元素
    last_elements = [triplet[2] for triplet in triplets]
    
    # 计算频率
    counter = Counter(last_elements)
    
    # 找出唯一出现的最后一个元素
    unique_last_elements = {element for element, count in counter.items() if count == 1}
    
    # 筛选出最后一个元素在唯一出现的三元组
    unique_triplets = [triplet for triplet in triplets if triplet[2] in unique_last_elements]
    
    return unique_triplets



def ReLinkByAFLinkWithlessCalculate2(MHTaddCluster,Cluster,frame):
    VisID(Cluster, frame)
    # HalfcostList = [] 
    costList =  LinkAWithB(MHTaddCluster,Cluster)

    # for ConncetPair1 in costList:
    #         if ConncetPair1[1] >  ConncetPair1[0]:
    #             HalfcostList.append(ConncetPair1)
    
    sorted_HalfcostList = sorted(costList, key=lambda x: x[2], reverse=True)

    sorted_HalfcostList=sorted_HalfcostList[:int(0.2*len(sorted_HalfcostList))]
    
    sorted_HalfcostListAboveMean,mean = filter_triplets(sorted_HalfcostList)

    # unique_triplets = unique_last_triplets(sorted_HalfcostListAboveMean)

    forest,ClusterFroset2 = MHRelinkV2(sorted_HalfcostListAboveMean,MHTaddCluster,Cluster,thr = 45)
    
    VisID(forest, frame)
    
    return forest,ClusterFroset2


def ReLinkByAFLinkWithlessCalculate3(MHTaddCluster,Cluster, frame):

    HalfcostList = [] 
    costList =  LinkAWithB(MHTaddCluster,Cluster)

    for ConncetPair1 in costList:
            if ConncetPair1[1] >  ConncetPair1[0]:
                HalfcostList.append(ConncetPair1)
    
    sorted_HalfcostList = sorted(HalfcostList, key=lambda x: x[2], reverse=True)


    if frame <=3:
        sorted_HalfcostList=sorted_HalfcostList[:int(0.5*len(sorted_HalfcostList))]

        # sorted_HalfcostListAboveMean,mean = filter_triplets(sorted_HalfcostList)

        # unique_triplets = unique_last_triplets(sorted_HalfcostListAboveMean)
                    
        forest,ClusterFroset2 = MHRelinkV2(sorted_HalfcostList,MHTaddCluster,Cluster,thr = 50)
    
    else:    
        # sorted_HalfcostList=sorted_HalfcostList[:int(0.4*len(sorted_HalfcostList))]

        # sorted_HalfcostListAboveMean,mean = filter_triplets(sorted_HalfcostList)

        # unique_triplets = unique_last_triplets(sorted_HalfcostListAboveMean)
        # non_integer_triples = []

        # # 判断并保存
        # for triple in sorted_HalfcostListAboveMean:
        #     last_element = triple[2]
        #     if isinstance(last_element, float) and last_element % 1 != 0:
        #         non_integer_triples.append(triple)
                    
        forest,ClusterFroset2 = MHRelinkV2(sorted_HalfcostList,MHTaddCluster,Cluster,thr = 50)

        
        VisID(forest, frame)


    return forest,ClusterFroset2