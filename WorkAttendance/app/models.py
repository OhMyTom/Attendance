from app import db
import numpy as np
import zlib
import io


class Worker(db.Model):
    __tablename__ = 'worker_info'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    face_feature = db.Column(db.LargeBinary, nullable=False)

    @staticmethod
    def adapt_array(arr):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return zlib.compress(out.read())

    @staticmethod
    def convert_array(text):
        out = io.BytesIO(zlib.decompress(text))
        return np.load(out)


class Attendance(db.Model):
    __tablename__ = 'logcat'
    id = db.Column(db.Integer, primary_key=True)
    worker_id = db.Column(db.Integer, db.ForeignKey('worker_info.id'), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    datetime = db.Column(db.String(50), nullable=False)
    late = db.Column(db.String(10), nullable=False)
    date = db.Column(db.String(10), nullable=False)

    worker = db.relationship('Worker', backref=db.backref('attendances', lazy=True))