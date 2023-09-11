from enum import StrEnum, auto

class VideoExtensions(StrEnum):
    # others exist... "ffmpeg -muxers"
    AVI = auto()
    FLV = auto()
    M4V = auto()
    MKV = auto()
    MOV = auto()
    MP4 = auto()
    WEBM = auto()