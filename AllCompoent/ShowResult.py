from tools.settings import *

#目标关联
def FindMatchTargets(all_data,FrameMatch,NextFrameTargetToBeMatched):
     for data in all_data:
          if(data["frame"] == NextFrameTargetToBeMatched):
            targets = data["targets"]
            for target in targets:
                if( target.id == FrameMatch):
                    return target

def FindTheTrajectory(trajectory_forest, id):
    for trajectory in trajectory_forest:
        if trajectory.targets[-1].id == id:
            return trajectory

#图匹配结果展示
def combine_images(image1, image2, orientation='horizontal'):
    # 确保两张图片有相同的高度（水平合并）或宽度（垂直合并）
    if orientation == 'horizontal':
        if image1.shape[0] != image2.shape[0]:
            raise ValueError("Error: Images must have the same height to be combined horizontally.")
        combined_image = cv2.hconcat([image1, image2])
    elif orientation == 'vertical':
        if image1.shape[1] != image2.shape[1]:
            raise ValueError("Error: Images must have the same width to be combined vertically.")
        combined_image = cv2.vconcat([image1, image2])
    else:
        raise ValueError("Error: Orientation must be either 'horizontal' or 'vertical'.")
    return combined_image

def string_to_array(input_string):
    # 使用逗号分隔字符串并去除可能的前后空格
    array = [int(item.strip()) for item in input_string.split(',')]
    return array

def read_specific_line(file_path, line_number):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        return lines[line_number]

def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

def ProducePairRes(CurrentFrame,PreviousFrame, GraphMatch, all_data, yz, Seqlist):
    ListPicture2 = 'D:\\PaperForReId\\GMOT40\\GMOT40\\GenericMOT_JPEG_Sequence\\' + str(Seqlist) + '\\img1\\'

    LeftPicture2Path=ListPicture2 + str(PreviousFrame).zfill(6) + ".jpg"
    RightPicturePath=ListPicture2 + str(CurrentFrame).zfill(6) + ".jpg"
    
    image1 = cv2.imread(LeftPicture2Path)
    image2 = cv2.imread(RightPicturePath)

    x_values = [x for x, y in GraphMatch]
    y_values = [y for x, y in GraphMatch]

    PreviousFrameQD = []
    CurrentFrameQD = []

    for data in all_data:
        if PreviousFrame == data["frame"]:
            for target in  data["targets"]:
                if target.id not in x_values :
                    PreviousFrameQD.append(target.id)
    
    for data in all_data:
        if CurrentFrame == data["frame"]:
            for target in  data["targets"]:
                if target.id not in y_values :
                    CurrentFrameQD.append(target.id)

            

    combined_image = combine_images(image1, image2, orientation='horizontal')

    for EveryPairs in GraphMatch:
        LeftPiccontent = FindMatchTargets(all_data, EveryPairs[0], PreviousFrame)
        RightPiccontent = FindMatchTargets(all_data, EveryPairs[1], CurrentFrame)
        # print(LeftPiccontent)
        # print(RightPiccontent)

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        center1=(int(LeftPiccontent.x + 0.5*LeftPiccontent.width),int(LeftPiccontent.y+0.5*LeftPiccontent.height))
        cv2.circle(combined_image, center1, 3, color, thickness=-1)
        cv2.putText(combined_image, str(LeftPiccontent.id), center1, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
        
        py = int(0.5*combined_image.shape[1])
        center2=(int(RightPiccontent.x+0.5*RightPiccontent.width) + py ,int(RightPiccontent.y+0.5*RightPiccontent.height))
        
        cv2.circle(combined_image, center2, 3, color, thickness=-1)
        cv2.putText(combined_image, str(RightPiccontent.id), center2, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)

        cv2.line(combined_image, center1, center2, color, thickness=2)
    
    for index in PreviousFrameQD:
         LeftPiccontent = FindMatchTargets(all_data, index, PreviousFrame)
         center1=(int(LeftPiccontent.x + 0.5*LeftPiccontent.width),int(LeftPiccontent.y+0.5*LeftPiccontent.height))
         cv2.circle(combined_image, center1, 3, (0,0,0), thickness=-1)
         top_left = (int(LeftPiccontent.x ),int(LeftPiccontent.y))
         bottom_right = (int(LeftPiccontent.x + 1*LeftPiccontent.width),int(LeftPiccontent.y + 1*LeftPiccontent.height))
         cv2.rectangle(combined_image, top_left, bottom_right, (0,0,0), thickness = 3)
         cv2.putText(combined_image, str(LeftPiccontent.id), bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4, cv2.LINE_AA)
        
    
    for index in CurrentFrameQD:
        RightPiccontent = FindMatchTargets(all_data, index, CurrentFrame)
        py = int(0.5*combined_image.shape[1])
        center2=(int(RightPiccontent.x+0.5*RightPiccontent.width) + py ,int(RightPiccontent.y+0.5*RightPiccontent.height))
        cv2.circle(combined_image, center1, 3, (0,0,0), thickness=-1)
        top_left = (int(RightPiccontent.x ) + py , int(RightPiccontent.y))
        bottom_right = (int(RightPiccontent.x + 1*RightPiccontent.width) + py, int(RightPiccontent.y + 1*RightPiccontent.height))
        cv2.rectangle(combined_image, top_left, bottom_right, (0,0,0), thickness = 3)
        cv2.putText(combined_image, str(RightPiccontent.id), bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4, cv2.LINE_AA)
    
    if not os.path.exists('D://PaperForReId/MHTTrack/NEWresult/' + str(Seqlist)):
        # 创建文件夹
        os.makedirs('D://PaperForReId/MHTTrack/NEWresult/' + str(Seqlist))
        print(f"文件夹'{Seqlist}'已创建。")
    else:
        print(f"文件夹'{Seqlist}'已存在。")

    output_path = 'D://PaperForReId/MHTTrack/NEWresult/'+ str(Seqlist) +'/'+ str(Seqlist)+ "_" + str(PreviousFrame) + "_" + str(CurrentFrame)+ "_" +str(yz)+".jpg"
    save_image(combined_image,output_path)

def ProducePairResV2(CurrentFrame,PreviousFrame, GraphMatch, all_data, yz, Seqlist):
    ListPicture2 = 'D:\PaperForReId\MHTTrack\FakeData\\'

    LeftPicture2Path=ListPicture2 + str(PreviousFrame).zfill(6) + ".jpg"
    RightPicturePath=ListPicture2 + str(CurrentFrame).zfill(6) + ".jpg"
    # path='D:\PaperForReId\MHTTrack\FakeData\\000000.jpg'
    # print(path)  
    # image1 = cv2.imread(path)
    # print(LeftPicture2Path)
    image1 = cv2.imread(LeftPicture2Path)
    image2 = cv2.imread(RightPicturePath)

    x_values = [x for x, y in GraphMatch]
    y_values = [y for x, y in GraphMatch]

    PreviousFrameQD = []
    CurrentFrameQD = []

    for data in all_data:
        if PreviousFrame == data["frame"]:
            for target in  data["targets"]:
                if target.id not in x_values :
                    PreviousFrameQD.append(target.id)
    
    for data in all_data:
        if CurrentFrame == data["frame"]:
            for target in  data["targets"]:
                if target.id not in y_values :
                    CurrentFrameQD.append(target.id)

            

    combined_image = combine_images(image1, image2, orientation='horizontal')

    for EveryPairs in GraphMatch:
        LeftPiccontent = FindMatchTargets(all_data, EveryPairs[0], PreviousFrame)
        RightPiccontent = FindMatchTargets(all_data, EveryPairs[1], CurrentFrame)
        # print(LeftPiccontent)
        # print(RightPiccontent)

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        center1=(int(LeftPiccontent.x + 0.5*LeftPiccontent.width),int(LeftPiccontent.y+0.5*LeftPiccontent.height))
        cv2.circle(combined_image, center1, 3, color, thickness=-1)
        cv2.putText(combined_image, str(LeftPiccontent.id), center1, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
        
        py = int(0.5*combined_image.shape[1])
        center2=(int(RightPiccontent.x+0.5*RightPiccontent.width) + py ,int(RightPiccontent.y+0.5*RightPiccontent.height))
        
        cv2.circle(combined_image, center2, 3, color, thickness=-1)
        cv2.putText(combined_image, str(RightPiccontent.id), center2, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)

        cv2.line(combined_image, center1, center2, color, thickness=2)
    
    for index in PreviousFrameQD:
         LeftPiccontent = FindMatchTargets(all_data, index, PreviousFrame)
         center1=(int(LeftPiccontent.x + 0.5*LeftPiccontent.width),int(LeftPiccontent.y+0.5*LeftPiccontent.height))
         cv2.circle(combined_image, center1, 3, (0,0,0), thickness=-1)
         top_left = (int(LeftPiccontent.x ),int(LeftPiccontent.y))
         bottom_right = (int(LeftPiccontent.x + 1*LeftPiccontent.width),int(LeftPiccontent.y + 1*LeftPiccontent.height))
         cv2.rectangle(combined_image, top_left, bottom_right, (0,0,0), thickness = 3)
         cv2.putText(combined_image, str(LeftPiccontent.id), bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4, cv2.LINE_AA)
        
    
    for index in CurrentFrameQD:
        RightPiccontent = FindMatchTargets(all_data, index, CurrentFrame)
        py = int(0.5*combined_image.shape[1])
        center2=(int(RightPiccontent.x+0.5*RightPiccontent.width) + py ,int(RightPiccontent.y+0.5*RightPiccontent.height))
        cv2.circle(combined_image, center1, 3, (0,0,0), thickness=-1)
        top_left = (int(RightPiccontent.x ) + py , int(RightPiccontent.y))
        bottom_right = (int(RightPiccontent.x + 1*RightPiccontent.width) + py, int(RightPiccontent.y + 1*RightPiccontent.height))
        cv2.rectangle(combined_image, top_left, bottom_right, (0,0,0), thickness = 3)
        cv2.putText(combined_image, str(RightPiccontent.id), bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4, cv2.LINE_AA)
    
    if not os.path.exists('D://PaperForReId/MHTTrack/NEWresult/' + 'FakaData'):
        # 创建文件夹
        os.makedirs('D://PaperForReId/MHTTrack/NEWresult/' + 'FakaData')
        print(f"文件夹'{Seqlist}'已创建。")
    else:
        print(f"文件夹'{Seqlist}'已存在。")

    output_path = 'D://PaperForReId/MHTTrack/NEWresult/'+ 'FakaData' + '/'+ 'FakaData'+ "_" + str(PreviousFrame) + "_" + str(CurrentFrame)+ "_" +str(yz)+".jpg"
    save_image(combined_image,output_path)


#可视化
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
