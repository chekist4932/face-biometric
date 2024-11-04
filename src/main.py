from fastapi import FastAPI

from src.face_auth.router import router as face_router
from src.exceptions import register_exception_handlers

app = FastAPI(title='face-biometric')

register_exception_handlers(app)

app.include_router(face_router)
