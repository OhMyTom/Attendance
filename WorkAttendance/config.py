import os
from dotenv import load_dotenv

load_dotenv()

# 百度API配置
BAIDU_API_KEY = "bjZTVrTLCVlsRAMarsFug6RQ"
BAIDU_SECRET_KEY = "WeXLvrskcKaxmNooajVLJN4v9cnO70LA"

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///attendance.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 百度API配置
    BAIDU_API_KEY = BAIDU_API_KEY
    BAIDU_SECRET_KEY = BAIDU_SECRET_KEY