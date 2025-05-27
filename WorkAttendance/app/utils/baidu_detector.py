import requests
import json
import cv2
import base64
import time
import numpy as np
from flask import current_app

# 百度API配置
API_KEY = "bjZTVrTLCVlsRAMarsFug6RQ"
SECRET_KEY = "WeXLvrskcKaxmNooajVLJN4v9cnO70LA"

class BaiduLivenessDetector:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
        self.ACCESS_TOKEN = self.get_access_token()
        self.token_expire_time = time.time() + 2592000 - 60  # 30天有效
        self.face_field = "quality,face_shape,face_type,spoofing"
        self.liveness_type = "NORMAL"
        self.quality_control = "NORMAL"
        self.action_type = "APP"
        self.action_sequence = []
        self.required_actions = ['blink', 'mouth', 'head']
            
    def init_app(self, app):
        """初始化应用配置"""
        pass

    def get_access_token(self):
        """获取百度API访问令牌"""
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": API_KEY,
            "client_secret": SECRET_KEY
        }
        response = requests.post(url, params=params)
        return response.json().get("access_token")

    def detect_liveness(self, frame):
        """检测活体"""
        if not self.ACCESS_TOKEN or time.time() > self.token_expire_time:
            self.ACCESS_TOKEN = self.get_access_token()
            self.token_expire_time = time.time() + 2592000 - 60

        # 图像转base64
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        url = f"https://aip.baidubce.com/rest/2.0/face/v3/faceverify?access_token={self.ACCESS_TOKEN}"
        headers = {'Content-Type': 'application/json'}
        data = [{
            "image": img_base64,
            "image_type": "BASE64",
            "face_field": self.face_field,
            "liveness_control": self.liveness_type,
            "quality_control": self.quality_control
        }]
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            result = response.json()
            print("百度API返回：", result)
            if result.get("error_code") == 0:
                # 1. 优先用face_liveness字段
                liveness_score = result.get("result", {}).get("face_liveness", 0)
                # 2. 如果没有face_liveness字段，再用face_list[0]["liveness"]["livemapscore"]
                if not liveness_score:
                    face_list = result.get("result", {}).get("face_list", [])
                    if face_list:
                        liveness_score = face_list[0].get("liveness", {}).get("livemapscore", 0)
                print("最终用于判断的liveness_score:", liveness_score)
                if liveness_score > 0.7:
                    return True, "活体检测通过"
            return False, "活体检测未通过"
        except Exception as e:
            print("活体检测失败:", e)
            return False, f"活体检测失败: {str(e)}"

    def reset(self):
        """重置检测器状态"""
        self.action_sequence = [] 