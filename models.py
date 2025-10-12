# src/models.py

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users_main'

    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    plan = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def __repr__(self):
        return f"<User {self.company_name} - {self.email}>"
