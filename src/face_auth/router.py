from fastapi import APIRouter, UploadFile, File

from src.face_auth.service import  get_pswd

router = APIRouter(prefix='/api', tags=[''])


@router.post('/face')
async def pswd_by_face(file: UploadFile = File(...)):
    return await get_pswd(file)

