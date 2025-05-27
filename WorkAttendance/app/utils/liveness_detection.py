import cv2
import numpy as np
import dlib
import random
import os
from scipy.spatial import distance as dist
from flask import current_app

class LivenessDetector:
    def __init__(self):
        # 获取模型文件的绝对路径
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                'model', 
                                'shape_predictor_68_face_landmarks.dat')
        
        # 初始化dlib的人脸检测器和关键点预测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        
        # 定义眼睛和嘴巴的关键点索引
        self.EYE_AR_THRESH = 0.25  # 眼睛纵横比阈值
        self.MOUTH_AR_THRESH = 0.3  # 嘴巴纵横比阈值
        self.HEAD_POSE_THRESH = 20.0  # 头部姿态角度阈值
        
        # 眨眼计数器
        self.blink_counter = 0
        self.blink_threshold = 2  # 需要检测到的眨眼次数
        
        # 动作序列
        self.action_sequence = []
        self.available_actions = ['blink', 'mouth', 'head']  # 可用的动作
        self.required_actions = []  # 随机选择的两个动作
        self.action_prompts = {
            'blink': '请眨眼两次',
            'mouth': '请张嘴',
            'head': '请左右摇头'
        }
        
        # 初始化随机动作
        self._init_random_actions()
        
    def _init_random_actions(self):
        """初始化随机动作组合"""
        self.required_actions = random.sample(self.available_actions, 2)
        self.action_sequence = []
        self.blink_counter = 0
        
    def get_required_actions_prompt(self):
        """获取当前需要完成的动作提示"""
        prompts = [self.action_prompts[action] for action in self.required_actions]
        return '，'.join(prompts)
        
    def get_eye_aspect_ratio(self, eye):
        """计算眼睛纵横比"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
        
    def get_mouth_aspect_ratio(self, mouth):
        """计算嘴巴纵横比"""
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])
        mar = (A + B + C) / (2.0 * D)
        return mar
        
    def get_head_pose(self, shape):
        """计算头部姿态"""
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                               shape[39], shape[42], shape[45], shape[31], shape[35],
                               shape[48], shape[54], shape[57], shape[8]])
        
        model_pts = np.float32([[0.0, 0.0, 0.0],
                               [0.0, -330.0, -65.0],
                               [-225.0, 170.0, -135.0],
                               [225.0, 170.0, -135.0],
                               [-150.0, -150.0, -125.0],
                               [150.0, -150.0, -125.0],
                               [-150.0, 150.0, -125.0],
                               [150.0, 150.0, -125.0],
                               [0.0, 0.0, 0.0],
                               [0.0, -330.0, -65.0],
                               [-225.0, 170.0, -135.0],
                               [225.0, 170.0, -135.0],
                               [-150.0, -150.0, -125.0],
                               [150.0, -150.0, -125.0]])
        
        size = (640, 480)
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4,1))
        success, rotation_vec, translation_vec = cv2.solvePnP(model_pts, image_pts, camera_matrix, dist_coeffs)
        
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        
        return euler_angle
        
    def detect_blink(self, shape):
        """检测眨眼"""
        left_eye = np.array([(shape[36+i][0], shape[36+i][1]) for i in range(6)])
        right_eye = np.array([(shape[42+i][0], shape[42+i][1]) for i in range(6)])
        
        left_ear = self.get_eye_aspect_ratio(left_eye)
        right_ear = self.get_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < self.EYE_AR_THRESH:
            self.blink_counter += 1
            if self.blink_counter >= self.blink_threshold:
                if 'blink' not in self.action_sequence:
                    self.action_sequence.append('blink')
                return True
        else:
            self.blink_counter = 0
        return False
        
    def detect_mouth_movement(self, shape):
        """检测嘴部动作"""
        mouth = np.array([(shape[48+i][0], shape[48+i][1]) for i in range(20)])
        mar = self.get_mouth_aspect_ratio(mouth)
        
        if mar > self.MOUTH_AR_THRESH:
            if 'mouth' not in self.action_sequence:
                self.action_sequence.append('mouth')
            return True
        return False
        
    def detect_head_movement(self, shape):
        """检测头部动作"""
        euler_angle = self.get_head_pose(shape)
        
        if abs(euler_angle[0, 0]) > self.HEAD_POSE_THRESH or abs(euler_angle[1, 0]) > self.HEAD_POSE_THRESH:
            if 'head' not in self.action_sequence:
                self.action_sequence.append('head')
            return True
        return False
        
    def detect_liveness(self, frame):
        """综合检测活体"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return False, "未检测到人脸"
            
        shape = self.predictor(gray, faces[0])
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        # 执行各种检测
        self.detect_blink(shape)
        self.detect_mouth_movement(shape)
        self.detect_head_movement(shape)
        
        # 检查是否完成所有要求的动作
        if len(self.action_sequence) == len(self.required_actions):
            # 重置检测器状态
            self._init_random_actions()
            return True, "活体检测通过"
            
        # 返回当前进度
        completed_actions = len(self.action_sequence)
        total_actions = len(self.required_actions)
        return False, f"已完成 {completed_actions}/{total_actions} 个动作"
        
    def reset(self):
        """重置检测器状态"""
        self._init_random_actions()

    def process_frame(self, frame):
        """处理视频帧"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return frame
        
        shape = self.predictor(gray, faces[0])
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.predictor(roi_gray, dlib.rectangle(x, y, x+w, y+h))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        
        return frame

    def get_random_actions(self):
        """获取两个随机动作"""
        self.current_actions = random.sample(self.available_actions, 2)
        self.action_status = {action: False for action in self.current_actions}
        return self.current_actions
