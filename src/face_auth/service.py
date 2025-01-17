import hashlib

from fastapi import UploadFile

from src.face_auth.utils.utils import predicate, bin_tens_to_hex


async def get_pswd(file: UploadFile):
    content = await file.read()
    predict = predicate(content)
    primary = bin_tens_to_hex(predict[0])
    pswd = hashlib.sha512(bytes.fromhex(primary)).hexdigest()

    return pswd
