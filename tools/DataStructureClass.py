
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import contextlib
import io
import math

def Cosine_Similarity_Matrix(vectors):
    """
    计算向量矩阵中所有向量两两之间的余弦相似度

    Args:
        vectors (numpy.ndarray): 包含向量的矩阵，每行为一个向量

    Returns:
        numpy.ndarray: 余弦相似度矩阵，大小为 (n, n)，n 是向量个数
    """
    # 计算每个向量的范数
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # 计算内积
    dot_products = np.dot(vectors, vectors.T)

    # 计算余弦相似度矩阵
    cosine_similarities = dot_products / (norms * norms.T)

    return cosine_similarities



class Nstr:
    def __init__(self, arg):
       self.x=arg
    def __sub__(self,other):
        c = self.x.replace(other.x,"")
        return c

class Target:
    def __init__(self, camera, frame, tid, x, y, width, height, confidence, appearance=None):
        self.camera=camera
        self.frame = frame
        self.id = tid
        self.sid = str(self.frame) + '-' + str(self.id)
        self.bid = -1
        self.gtid = -1
        self.x = x
        self.y = y
        self.position = np.array([x, y])       
        self.width = width
        self.height = height
        self.bbox = np.array([x, y, width, height])
        self.confidence = confidence
        self.appearance = np.array(appearance)
        
    def obj2dict(self):
        return {
            "frame": self.frame,
            "id": self.id,
        }

#拟合曲线表达式
def fit_curve_and_calculate_total_curvature(x, y, degree=2):
    """
    拟合一条曲线并计算其总曲率

    参数：
    x (array-like): x 坐标数组
    y (array-like): y 坐标数组
    degree (int): 多项式拟合的次数，默认是二次多项式

    返回：
    total_curvature (float): 拟合曲线的总曲率
    """
    # 多项式拟合
    
  
    popt = np.polyfit(x, y, degree)
    poly_func = np.poly1d(popt)
    
    # 计算曲率
    def calculate_curvature(poly, x):
        # 一阶导数和二阶导数
        first_derivative = np.polyder(poly, 1)
        second_derivative = np.polyder(poly, 2)
        # 计算曲率
        return np.abs(second_derivative(x)) / (1 + first_derivative(x)**2)**(3/2)
    
    # 曲率函数
    curvature_func = lambda x: calculate_curvature(poly_func, x)
    
    # 计算总曲率
    total_curvature, _ = quad(curvature_func, min(x), max(x))
    return total_curvature

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

class Trajectory:
    
    def __init__(self, targets, branch_id):
        if type(targets) == list:
           self.targets = targets
        else:
            self.targets = [targets]
        self.branch_id = branch_id
       
        self.tracklet=[]
    
    def GetTrajectoryLength(self):
        return self.targets[-1].frame - self.targets[0].frame + 1
    
    def GetTrajectoryWeight_IOU(self):
        if len(self.targets) is 1:
            return [1, -1, -1]
        SumOfIOU = 1

        for target in self.targets: 
            if self.targets.index(target) + 1 < len(self.targets):
                Nexttarget = self.targets[self.targets.index(target)+1]
                X_IL = max(target.x, Nexttarget.x)
                X_IR = min(target.x + target.width , Nexttarget.x + Nexttarget.width)
                Y_IL = max(target.y + target.height , Nexttarget.y + Nexttarget.height)
                Y_IU = min(target.y, Nexttarget.y)
                S_inter = (X_IR - X_IL)*(Y_IL - Y_IU)
                S_union = target.width * target.height + Nexttarget.width * Nexttarget.height - S_inter
                IOU = S_inter *1.0 / S_union
                SumOfIOU = SumOfIOU + IOU
            else:
                continue

        return [SumOfIOU ** 2, -1, -1]
    
    def GetTrajectoryWeight_FIT(self):
        min_length = 2

        if len(self.targets) < min_length:
            return 1
            # return len(self.targets)

        def linear(x, a, b):
            return a * x + b

        def extract_points(targets):
            X_axis = []
            Y_axis = []
            for target in targets:
                TargetCenterX = target.x + 0.5 * target.width
                TargetCenterY = target.y + 0.5 * target.height
                X_axis.append(TargetCenterX)
                Y_axis.append(TargetCenterY)
            return X_axis, Y_axis

        def fit_linear(X_axis, Y_axis):
            params, params_covariance = curve_fit(linear, X_axis, Y_axis)
            fitted_Y = linear(np.array(X_axis), *params)
            fitting_error = np.sqrt(np.mean((np.array(Y_axis) - fitted_Y) ** 2))
            return fitting_error

        n = len(self.targets)
        group_size = max(n // min_length, min_length)  # 确定组大小至少为5
        groups = [self.targets[i:i + group_size] for i in range(0, n, group_size)]

        if len(groups[-1]) < min_length:
            groups[-1] = self.targets[-1 * min_length:]

        errors = []
        for group in groups:
            X_axis, Y_axis = extract_points(group)
            error = fit_linear(X_axis, Y_axis)
            errors.append(error)

        # 计算平均误差
        average_error = len(self.targets) * math.exp(np.average(errors) * -1) + 1
        return average_error * average_error
    
    def GetTrajectoryWeight_DIS(self):
        min_length = 5
        frame_length = self.GetTrajectoryLength()

        if frame_length < min_length:
            return [frame_length ** 2, -1, -1]
   
        data = []
        for target in self.targets:
            TargetCenterX = target.x + 0.5 * target.width
            TargetCenterY = target.y + 0.5 * target.height
            data.append({"x": TargetCenterX, "y": TargetCenterY, "frame": target.frame})
        import statistics
        dX_axis=[]
        dY_axis=[]
        for index in range(len(data)-1):
            frame_gap = abs(data[index + 1]["frame"] - data[index]["frame"])
            dX_axis.append(abs(data[index]["x"] - data[index + 1]["x"]) / frame_gap) 
            dY_axis.append(abs(data[index]["y"] - data[index + 1]["y"]) / frame_gap) 

        
        Xvariance = statistics.pvariance(dX_axis)
        Yvariance = statistics.pvariance(dY_axis)
        return  [(frame_length + (math.exp(-1 * Xvariance - 1 * Yvariance))) ** 2, Xvariance, Yvariance]
    
    def GetTrajectoryWeight_DIS_Continue(self,Curframe):
        min_length = 5
        frame_length = self.GetTrajectoryLength()
        target_count = len(self.targets)

        # if target_count <= 3:
        #     # return [target_count**2, frame_length, -1, -1,-1,-1,-1,-1]
        #     return [(target_count) ** 2,target_count,Curframe + 1,-1,-1,-1,-1,-1]


        Continues = []
        for target in self.targets:
            Continues.append(target.frame)
        
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
        
        if frame_length < min_length:
            return [ (target_count + ContinuesWeight)** 2, -1, -1, ContinuesWeight,-1,-1,-1,-1]
   
        data = []
        for target in self.targets:
            TargetCenterX = target.x + 0.5 * target.width
            TargetCenterY = target.y + 0.5 * target.height
            data.append({"x": TargetCenterX, "y": TargetCenterY, "frame": target.frame})
        import statistics
        dX_axis=[]
        dY_axis=[]
        for index in range(len(data)-1):
            frame_gap = abs(data[index + 1]["frame"] - data[index]["frame"])
            dX_axis.append(abs(data[index]["x"] - data[index + 1]["x"]) / frame_gap) 
            dY_axis.append(abs(data[index]["y"] - data[index + 1]["y"]) / frame_gap) 

        
        dX2_axis=[]
        dY2_axis=[]
        for index in range(len(dX_axis)-1):
            dX2_axis.append(abs(dX_axis[index] - dX_axis[index+1])) 
            dY2_axis.append(abs(dY_axis[index] - dY_axis[index+1])) 
 
        Xvariance = statistics.pvariance(dX_axis) / target_count
        Yvariance = statistics.pvariance(dY_axis) / target_count

        # Xvariance2 = statistics.pvariance(dX2_axis) / target_count
        # Yvariance2 = statistics.pvariance(dY2_axis) / target_count

        w1 = 1 / math.log10((Xvariance + Yvariance) + 2)
        # w2 = 1 / math.log10((Xvariance2 + Yvariance2) + 2)

        # return  [(target_count + w1 + w2)**2, target_count, w1, w2,Xvariance,Xvariance2,Yvariance,Yvariance2]
        # return  [(frame_length / (Curframe + 1) * (math.exp(-1 * Xvariance ) + math.exp(-1 * Xvariance2 ) + math.exp(-1 * Yvariance )+ math.exp(-1 * Yvariance2 )) ) ** 2 \
                #  ,math.exp(-1 * Xvariance ), math.exp(-1 * Xvariance2 ),math.exp(-1 * Yvariance ),math.exp(-1 * Yvariance2 ),target_count / (Curframe + 1),Yvariance2, -1]
        return  [(target_count + ContinuesWeight + w1 )**2, -1, -1, target_count/(Curframe + 1) ,Xvariance, Yvariance, -1, -1]


    def GetTrajectoryWeight_AccandVecinFit(self,Curframe):
        min_length = 5
        frame_length = self.GetTrajectoryLength()
        target_count = len(self.targets)

        if target_count <= 3:
            # return [target_count**2, frame_length, -1, -1,-1,-1,-1,-1]
            return [(target_count/(Curframe + 1)) ** 2,target_count,Curframe + 1,-1,-1,-1,-1,-1]


        # Continues = []
        # for target in self.targets:
        #     Continues.append(target.frame)
        
        # max_length = 1
        # current_length = 1
        # for i in range(1, len(Continues)):
        #     if Continues[i] == Continues[i-1] + 1:
        #         current_length += 1
        #     else:
        #         if current_length > max_length:
        #             max_length = current_length
        #         current_length = 1
        
        # ContinuesWeight = max(current_length, max_length) 
        
        # ContinuesWeight = 0

        # if frame_length < min_length:
        #     return [ (frame_length + ContinuesWeight)** 2, -1, -1, ContinuesWeight]
   
        data = []
        for target in self.targets:
            TargetCenterX = target.x + 0.5 * target.width
            TargetCenterY = target.y + 0.5 * target.height
            data.append({"x": TargetCenterX, "y": TargetCenterY, "frame": target.frame})
        import statistics
        dX_axis=[]
        dY_axis=[]
        for index in range(len(data)-1):
            frame_gap = abs(data[index + 1]["frame"] - data[index]["frame"])
            dX_axis.append(abs(data[index]["x"] - data[index + 1]["x"]) / frame_gap) 
            dY_axis.append(abs(data[index]["y"] - data[index + 1]["y"]) / frame_gap) 

        
        dX2_axis=[]
        dY2_axis=[]
        for index in range(len(dX_axis)-1):
            dX2_axis.append(abs(dX_axis[index] - dX_axis[index+1])) 
            dY2_axis.append(abs(dY_axis[index] - dY_axis[index+1])) 
 
        Xvariance = statistics.pvariance(dX_axis) / target_count
        Yvariance = statistics.pvariance(dY_axis) / target_count

        Xvariance2 = statistics.pvariance(dX2_axis) / target_count
        Yvariance2 = statistics.pvariance(dY2_axis) / target_count

        w1 = 1 - math.log10(0.001 + (Xvariance + Yvariance) / 5)
        w2 = 1 - math.log10(0.001 + (Xvariance2 + Yvariance2) / 5)

        # return  [(target_count + w1 + w2)**2, target_count, w1, w2,Xvariance,Xvariance2,Yvariance,Yvariance2]
        # return  [(frame_length / (Curframe + 1) * (math.exp(-1 * Xvariance ) + math.exp(-1 * Xvariance2 ) + math.exp(-1 * Yvariance )+ math.exp(-1 * Yvariance2 )) ) ** 2 \
                #  ,math.exp(-1 * Xvariance ), math.exp(-1 * Xvariance2 ),math.exp(-1 * Yvariance ),math.exp(-1 * Yvariance2 ),target_count / (Curframe + 1),Yvariance2, -1]
        return  [(target_count / (Curframe + 1) * (math.exp(-1 * Xvariance -1 * Yvariance) + math.exp(-1 * Xvariance2 -1 * Yvariance2) ) ) ** 2 \
                 ,math.exp(-1 * Xvariance ), math.exp(-1 * Xvariance2 ),math.exp(-1 * Yvariance ),math.exp(-1 * Yvariance2 ),target_count / (Curframe + 1),Yvariance2, -1]
    
       
    
    # def GetTrajectoryWeight(self):
    #     #计算轨迹权重，利用轨迹长度/外观相似度/

    #     if(self.targets[0].appearance != None):
    #         if len(self.targets) < 2:
    #             return self.targets[0].confidence
    #         branch_appearances = np.array([target.appearance for target in self.targets]) 
    #         # 计算余弦相似度矩阵
    #         cosine_similarities = Cosine_Similarity_Matrix(branch_appearances)
    #         # 排除对角线上的相似度（每个向量与自身的相似度）
    #         np.fill_diagonal(cosine_similarities, 0)
    #         # 计算余弦相似度之和
    #         similarity_sum = np.sum(cosine_similarities) / 2
    #         return similarity_sum
        
    #     elif(self.targets[0].confidence != None):
    #         similarity_sum = 0.0 
    #         for target in self.targets: 
    #             similarity_sum += target.confidence
    #         return similarity_sum

    #     elif(False):#路径长度比
    #         if(len(self.targets) == 1):
    #             return 1
            
    #         trajectory = []
    #         for target in self.targets:
    #             TargetCenterX=target.x + 0.5* target.width
    #             TargetCenterY=target.y + 0.5* target.height
    #             trajectory.append((TargetCenterX,TargetCenterY))

    #         path_length = sum(euclidean_distance(trajectory[i], trajectory[i+1]) for i in range(len(trajectory) - 1))

    #         # Calculate the distance between the first and last points
    #         endpoint_distance = euclidean_distance(trajectory[0], trajectory[-1])
            
    #         # Calculate the Path Length Ratio (PLR)
    #         plr = path_length / endpoint_distance if endpoint_distance != 0 else np.inf
            
    #         if(plr==np.inf):
    #             return 0.5

    #         return plr + 1 

    #     elif(False):#曲率标准 
    #         if(len(self.targets) == 1):
    #             return 1
    #         if(len(self.targets) == 2):
    #             return 2
    #         BaseWeight = 2 
    #         X_axis=[]
    #         Y_axis=[]
    #         for target in self.targets:
    #             TargetCenterX=target.x + 0.5* target.width
    #             TargetCenterY=target.y + 0.5* target.height
    #             X_axis.append(TargetCenterX)
    #             Y_axis.append(TargetCenterY)
    #         f = io.StringIO()
    #         # 使用contextlib.redirect_stdout重定向标准输出到字符串IO对象
    #         with contextlib.redirect_stdout(f):
    #                 total_curvature = fit_curve_and_calculate_total_curvature(X_axis,Y_axis,degree=2)

    #         return BaseWeight + total_curvature

    #     else:#Iou标准
    #         if(len(self.targets) == 1):
    #             return 1
    #         SumOfIOU = 1
    #         for target in self.targets: 
    #             if self.targets.index(target) + 1 < len(self.targets):
    #                 Nexttarget = self.targets[self.targets.index(target)+1]
    #                 X_IL = max(target.x, Nexttarget.x)
    #                 X_IR = min(target.x + target.width , Nexttarget.x + Nexttarget.width)
    #                 Y_IL = max(target.y - target.height , Nexttarget.y - Nexttarget.height)
    #                 Y_IU = min(target.y, Nexttarget.y)
    #                 S_inter = (X_IR - X_IL)*(Y_IU - Y_IL)
    #                 S_union = target.width * target.height + Nexttarget.width * Nexttarget.height
    #                 IOU = S_inter *1.0 / S_union
    #                 SumOfIOU = SumOfIOU + IOU
    #             else:
    #                 continue
    #         return SumOfIOU
                

    
class FramePairs:
    def __init__(self,i,j,Matrix):
        self.i = i
        self.j = j
        self.Matrix = Matrix


    

def Xaverage(targets):
    x_coordinates = [target.x for target in targets]
    x_array = np.array(x_coordinates)
    average_x = np.mean(x_array)
    return average_x

def Yaverage(targets):
    y_coordinates = [target.x for target in targets]
    y_array = np.array(y_coordinates)
    average_y = np.mean(y_array)
    return average_y

def Confaverage(targets):
    if(targets==None):
        conf = [target.confidence for target in targets]
        confarray = np.array(conf)
        resulconf = np.mean(confarray)
        return resulconf
    else: return None

def APPaverage(targets):
    if(targets==None):
        appearances_list = [target.appearance for target in targets]
        appearances_array = np.array(appearances_list)
        mean_appearance = np.mean(appearances_array, axis=0)
        return mean_appearance
    else: return None


def Waverage(targets):
    width = [target.width for target in targets]
    widtharray = np.array(width)
    resultwidth = np.mean(widtharray)
    return resultwidth


def Haverage(targets):
    height = [target.width for target in targets]
    heightarray = np.array(height)
    resultheight = np.mean(heightarray)
    return resultheight


def FindFrame(targets):
    return targets[0].frame



def FindID(targets):
    idlist=[]
    for target in targets:
        idlist.append(target.id)
    return idlist



def Getsid(targets):
    return str(targets[0].frame) + '-' + str(targets[0].id)


class Tracklet:
    def __init__(self, targets):
        if type(targets) == list:
           self.targets = targets
        else:
            self.targets = [targets]

        self.frame = FindFrame(targets)
        self.ids = FindID(targets)
        self.id = self.ids[0]
        self.x = Xaverage(targets)
        self.y = Yaverage(targets)
        self.position = np.array([self.x, self.y]) 
        self.width = Waverage(targets)
        self.height = Haverage(targets)
        self.confidence = Confaverage(targets)
        self.appearance = APPaverage(targets)
        self.sid=Getsid(targets)
        self.bbox = np.array([Xaverage(targets), Yaverage(targets), Waverage(targets), Haverage(targets)])



class TrajectoryTracklet:
    
    def __init__(self, targets, branch_id):
       self.targets = targets
       self.branch_id = branch_id
       
       self.tracklet=[]
    
    def GetTrajectoryLength(self):
        return len(self.targets)
    
    def GetTrajectoryWeight(self):
        #计算轨迹权重，利用轨迹长度/外观相似度/
        if(self.targets[0].appearance != None):
            if len(self.targets) < 2:
                return self.targets[0].confidence
            branch_appearances = np.array([target.appearance for target in self.targets]) 

            # 计算余弦相似度矩阵
            cosine_similarities = Cosine_Similarity_Matrix(branch_appearances)

            # 排除对角线上的相似度（每个向量与自身的相似度）
            np.fill_diagonal(cosine_similarities, 0)

            # 计算余弦相似度之和
            similarity_sum = np.sum(cosine_similarities) / 2

            return similarity_sum
        else:
            similarity_sum = 0.0 
            for target in self.targets: 
                similarity_sum += target.conf
            
            return similarity_sum
    # def AddTracklet2Branch(self,target):



#可视化
import uuid
import hashlib
import random


def GetColorGMOT(id):
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


class VisualizeStructure:
    def __init__(self, target, trajectory):
        self.x = target.x
        self.y = target.y
        self.w = target.width
        self.h = target.height 
        self.id = target.id
        self.color_for_bbox = GetColorGMOT(trajectory.targets[0].sid.replace("-", ""))