from app import create_app, db
from app.models import Worker, Attendance

def init_db():
    app = create_app()
    with app.app_context():
        # 创建所有表
        db.create_all()
        print("数据库表创建成功！")

if __name__ == '__main__':
    init_db() 