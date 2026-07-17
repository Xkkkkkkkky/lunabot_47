import base64
import hashlib
import hmac
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


NAPCAT_STREAM_FILE_PREFIX = "napcat-stream://"
_REFERENCE_SECRET = os.urandom(32)


class NapCatStreamUploadError(RuntimeError):
    pass


def make_napcat_stream_file_ref(file_path: str) -> str:
    path = os.path.abspath(file_path)
    encoded_path = base64.urlsafe_b64encode(os.fsencode(path)).decode("ascii").rstrip("=")
    signature = hmac.new(
        _REFERENCE_SECRET,
        encoded_path.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    token = f"{encoded_path}.{signature}"
    return f"{NAPCAT_STREAM_FILE_PREFIX}{token}"


def parse_napcat_stream_file_ref(file_ref: str) -> str | None:
    if not isinstance(file_ref, str) or not file_ref.startswith(NAPCAT_STREAM_FILE_PREFIX):
        return None
    token = file_ref[len(NAPCAT_STREAM_FILE_PREFIX):]
    try:
        encoded_path, signature = token.rsplit(".", 1)
        encoded_path_bytes = encoded_path.encode("ascii")
    except (ValueError, UnicodeEncodeError):
        return None
    expected_signature = hmac.new(
        _REFERENCE_SECRET,
        encoded_path_bytes,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        return None
    encoded_path += "=" * (-len(encoded_path) % 4)
    try:
        return os.fsdecode(base64.urlsafe_b64decode(encoded_path.encode("ascii")))
    except Exception:
        return None


def _unwrap_api_response(response: Any) -> dict:
    if not isinstance(response, dict):
        return {}
    data = response.get("data")
    if response.get("status") == "ok" and isinstance(data, dict):
        return data
    return response


def _calculate_sha256(file_path: str, chunk_size: int) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


async def upload_file_via_napcat_stream(
    bot,
    file_path: str,
    *,
    chunk_size: int = 64 * 1024,
    file_retention_ms: int = 5 * 60 * 1000,
    verify_sha256: bool = True,
) -> str:
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"待上传文件不存在: {file_path}")
    if chunk_size <= 0:
        raise ValueError("NapCat Stream 分块大小必须大于 0")
    if file_retention_ms < 0:
        raise ValueError("NapCat Stream 文件保留时间不能小于 0")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise NapCatStreamUploadError(f"NapCat Stream 不支持上传空文件: {file_path}")

    stream_id = str(uuid4())
    total_chunks = math.ceil(file_size / chunk_size)
    suffix = Path(file_path).suffix
    remote_filename = f"lunabot_{stream_id}{suffix}"
    expected_sha256 = _calculate_sha256(file_path, chunk_size) if verify_sha256 else None

    try:
        with open(file_path, "rb") as f:
            for chunk_index in range(total_chunks):
                chunk = f.read(chunk_size)
                params = {
                    "stream_id": stream_id,
                    "chunk_data": base64.b64encode(chunk).decode("ascii"),
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "file_size": file_size,
                    "filename": remote_filename,
                    "file_retention": file_retention_ms,
                }
                if expected_sha256:
                    params["expected_sha256"] = expected_sha256
                response = await bot.call_api("upload_file_stream", **params)
                result = _unwrap_api_response(response)
                if result.get("status") not in ("chunk_received", "file_created"):
                    raise NapCatStreamUploadError(
                        f"NapCat Stream 分块 {chunk_index} 上传响应异常: {response}"
                    )

        response = await bot.call_api(
            "upload_file_stream",
            stream_id=stream_id,
            is_complete=True,
            file_retention=file_retention_ms,
        )
        result = _unwrap_api_response(response)
        remote_path = result.get("file_path")
        if result.get("status") != "file_complete" or not remote_path:
            raise NapCatStreamUploadError(f"NapCat Stream 文件合并响应异常: {response}")
        return str(remote_path)
    except Exception:
        try:
            await bot.call_api(
                "upload_file_stream",
                stream_id=stream_id,
                reset=True,
                file_retention=file_retention_ms,
            )
        except Exception:
            pass
        raise
