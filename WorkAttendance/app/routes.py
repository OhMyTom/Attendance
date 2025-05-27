from flask import Blueprint, render_template, jsonify, request, current_app
from app.models import Worker, Attendance, db
from app.utils.face_recognition import FaceRecognition
from app.utils.liveness_detection import LivenessDetector
from app.utils.baidu_detector import BaiduLivenessDetector
from datetime import datetime
import cv2
import numpy as np
import base64

main = Blueprint('main', __name__)
face_recognition = FaceRecognition()
local_detector = LivenessDetector()
baidu_detector = BaiduLivenessDetector()

def init_app(app):
    """初始化应用"""
    face_recognition.init_app(app)
    baidu_detector.init_app(app)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/get_actions', methods=['GET'])
def get_actions():
    """获取随机动作"""
    actions = local_detector.get_random_actions()
    return jsonify({
        'success': True,
        'actions': actions
    })

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.json
            image_data = data.get('image', '').split(',')[1]
            worker_id = data.get('worker_id')
            name = data.get('name')

            if not image_data or not worker_id or not name:
                 return jsonify({'success': False, 'message': '缺少必要的注册信息'})

            # 检查工号是否已存在
            if Worker.query.get(worker_id):
                return jsonify({'success': False, 'message': '工号已存在'})

            # 检查姓名是否已存在
            if Worker.query.filter_by(name=name).first():
                return jsonify({'success': False, 'message': '该姓名已被注册'})

            # 解码图片
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 获取人脸特征
            face_feature = face_recognition.get_face_feature(img)
            if face_feature is None:
                return jsonify({'success': False, 'message': '未检测到人脸'})

            # 检查是否与已有的人脸特征相似
            existing_workers = Worker.query.all()
            for worker in existing_workers:
                if worker.face_feature is not None:
                    similarity = face_recognition.compare_faces(face_feature, worker.face_feature)
                    if similarity > 0.6:  # 设置相似度阈值
                        return jsonify({
                            'success': False, 
                            'message': f'该人脸特征与已注册用户 {worker.name}({worker.id}) 相似度过高'
                        })

            # 保存到数据库
            worker = Worker(
                id=worker_id,
                name=name,
                face_feature=Worker.adapt_array(face_feature)
            )
            db.session.add(worker)
            db.session.commit()

            return jsonify({'success': True, 'message': '注册成功'})

        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    return render_template('register.html')

@main.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        try:
            data = request.json
            image_data = data.get('image', '').split(',')[1]
            use_liveness = data.get('use_liveness', False)
            liveness_type = data.get('liveness_type', 'local')

            if not image_data:
                return jsonify({'success': False, 'message': '未接收到图像数据'})

            # 解码图片
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 活体检测
            if use_liveness:
                if liveness_type == 'local':
                    success, message = local_detector.detect_liveness(img)
                else:  # baidu
                    success, message = baidu_detector.detect_liveness(img)
                    
                if not success:
                    return jsonify({'success': False, 'message': message})

            # 人脸识别
            worker = face_recognition.recognize_face(img)
            if not worker:
                return jsonify({'success': False, 'message': '未识别到人脸'})

            # 记录考勤
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H:%M:%S')

            # 检查是否已打卡
            existing = Attendance.query.filter_by(
                worker_id=worker.id,
                date=date_str
            ).first()

            if existing:
                existing.datetime = f"{date_str} {time_str}"
                existing.late = "是" if time_str > "09:00:00" else "否"
                db.session.commit()
                message = f"更新签到记录 - {worker.name}({worker.id})"
            else:
                attendance = Attendance(
                    worker_id=worker.id,
                    name=worker.name,
                    datetime=f"{date_str} {time_str}",
                    late="是" if time_str > "09:00:00" else "否",
                    date=date_str
                )
                db.session.add(attendance)
                db.session.commit()
                message = f"新增签到记录 - {worker.name}({worker.id})"

            return jsonify({
                'success': True,
                'message': message,
                'worker': {
                    'id': worker.id,
                    'name': worker.name
                }
            })

        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    # 获取考勤记录
    records = Attendance.query.order_by(Attendance.date.desc(), Attendance.datetime.desc()).all()
    return render_template('attendance.html', records=records)