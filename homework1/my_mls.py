import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class Deformation_parameters:
#获得变换后的各个参数，方便下面实现点的各个变换
    def __init__(self , control_point , target_point , alpha , eps):
        self.control_point = control_point
        self.target_point = target_point
        self.alpha = alpha
        self.eps = eps
    
    def get_weight_by_vector(self,point):
        #获取行向量形式的权重，令为行向量是为了能直接计算对应的点。
        n = self.control_point.shape[0]
        weight_vector = np.zeros(n)
        for i in range(n):
            s = np.linalg.norm(self.control_point[i,:] - point) + self.eps
            ss = s ** (2 * self.alpha)
            weight_vector[i] = 1/ss
        return weight_vector
    
    def get_point_center(self,point):
        #获得点的中心，并将结果存储在一个2x2的矩阵中，第一行是控制点的中心，第二个是目标点的中心。
        weight_vector = self.get_weight_by_vector(point)
        control_center = np.dot(weight_vector,self.control_point)/np.sum(weight_vector)
        target_center = np.dot(weight_vector,self.target_point)/np.sum(weight_vector)
        center = np.zeros((2,2))
        center[0,:] = control_center
        center[1,:] = target_center
        return center

#已经于是我们实际上获得了所需要的所有参数，于是下面便是实现变换的过程：

class Deformation:
    def __init__(self , control_point , target_point , alpha , eps) :
        self.control_point = control_point
        self.target_point = target_point
        self.alpha = alpha
        self.eps = eps
    
    #实现仿射变换，将点映成对应点
    def aff_def(self,point):
        deformation = Deformation_parameters(self.control_point,self.target_point,self.alpha,self.eps)
        weight_vector = deformation.get_weight_by_vector(point)
        center = deformation.get_point_center(point)
        #下面计算各个矩阵：
        fixed_control_point = self.control_point - center[0,:]
        fixed_target_point = self.target_point - center[1,:]
        weight_mat = np.diag(weight_vector)
        m0 = np.dot(fixed_control_point.T,weight_mat)
        m1 = np.dot(m0,fixed_control_point)
        if np.linalg.det(m1) < self.eps:
            affined_point = point
        else:
            m2 = np.dot(m0,fixed_target_point)
            inv_m1 = np.linalg.inv(m1)
            m2 = np.dot(inv_m1,m2)
            affined_point = np.dot((point - center[0,:] ),m2) + center[1,:]
        del deformation
        return affined_point
    
    #获取相似变换之后的点
    def sim_def(self,point):
        deformation = Deformation_parameters(self.control_point,self.target_point,self.alpha,self.eps)
        weight_vector = deformation.get_weight_by_vector(point)
        center = deformation.get_point_center(point)
        fixed_control_point = self.control_point - center[0,:]
        fixed_target_point = self.target_point - center[1,:]
        weight_mat = np.diag(weight_vector)
        m0 = np.dot(weight_mat,fixed_control_point)
        fixed_target_point_tran = fixed_target_point.T
        fixed_control_point_tran = fixed_control_point.T
        mt = np.array([[0,1],[-1,0]])
        m1 = np.dot(fixed_target_point_tran,m0)
        m2 = np.dot(m1,mt)
        m3 = np.dot(fixed_control_point_tran,m0)
        miu = np.trace(m3)
        if miu < self.eps:
            similared_point = point
        else:
            cos_the = np.trace(m1)
            sin_the = np.trace(m2)
            m = (cos_the * np.eye(2) + sin_the * mt) / miu
            similared_point = np.dot((point - center[0,:] ),m) + center[1,:]
        del deformation
        return similared_point
    
    #下面实现点的刚性变换
    def rig_def(self,point):
        deformation = Deformation_parameters(self.control_point,self.target_point,self.alpha,self.eps)
        weight_vector = deformation.get_weight_by_vector(point)
        center = deformation.get_point_center(point)
        fixed_control_point = self.control_point - center[0,:]
        fixed_target_point = self.target_point - center[1,:]
        weight_mat = np.diag(weight_vector)
        m0 = np.dot(weight_mat , fixed_control_point)
        fixed_target_point_tran = fixed_target_point.T
        mt = np.array([[0,1],[-1,0]])
        m1 = np.dot(fixed_target_point_tran,m0)
        m2 = np.dot(m1,mt)
        cos_the = np.trace(m1)
        sin_the = np.trace(m2)
        m3 = (cos_the * np.eye(2) + sin_the * mt) 
        miu = np.linalg.det(m3)
        if miu < self.eps:
            righted_point = point
        else:
            m = m3 / (miu ** 0.5)
            righted_point = np.dot((point - center[0,:] ),m) + center[1,:]
        del deformation
        return righted_point
