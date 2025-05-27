import os
import base64
import requests
import cv2
import numpy as np
import dlib
from flask import current_app

class FaceRecognition:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
            
        # 初始化 dlib 的人脸检测器和特征提取器
        self.detector = dlib.get_frontal_face_detector()
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                'model')
        self.predictor = dlib.shape_predictor(os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat'))
        self.face_rec_model = dlib.face_recognition_model_v1(os.path.join(model_path, 'dlib_face_recognition_resnet_model_v1.dat'))
            
    def init_app(self, app):
        """初始化应用配置"""
        self.app = app
        
    def get_face_feature(self, image):
        """获取人脸特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.detector(gray, 1)
        if len(faces) == 0:
            return None
            
        # 获取人脸关键点
        shape = self.predictor(gray, faces[0])
        
        # 计算人脸特征向量
        face_descriptor = self.face_rec_model.compute_face_descriptor(image, shape)
        
        # 转换为numpy数组
        face_feature = np.array(face_descriptor)
        return face_feature

    def recognize_face(self, image):
        """识别人脸"""
        from app.models import Worker
        
        # 获取当前图片的人脸特征
        current_feature = self.get_face_feature(image)
        if current_feature is None:
            return None
            
        # 在数据库中查找匹配的人脸
        workers = Worker.query.all()
        for worker in workers:
            # 将数据库中的人脸特征转换为numpy数组
            stored_feature = Worker.convert_array(worker.face_feature)
            
            # 计算欧氏距离
            distance = self.return_euclidean_distance(current_feature, stored_feature)
            
            # 如果距离小于阈值，认为是同一个人
            if distance == "same":
                return worker
                
        return None
        
    def return_euclidean_distance(self, feature_1, feature_2):
        """计算两个特征向量之间的欧氏距离"""
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        print("欧式距离: ", dist)
        if dist > 0.4:  # 阈值可以调整
            return "diff"
        else:
            return "same"

    def detect_face(self, image):
        """检测图片中的人脸"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        return len(faces) > 0

    def compare_faces(self, image1, image2):
        """比较两张人脸图片的相似度"""
        # 获取两张图片的人脸特征
        feature1 = self.get_face_feature(image1)
        feature2 = self.get_face_feature(image2)
        
        if feature1 is None or feature2 is None:
            return False
            
        # 计算欧氏距离
        result = self.return_euclidean_distance(feature1, feature2)
        return result == "same"

    @staticmethod
    def image_to_base64(image):
        """将图片转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
