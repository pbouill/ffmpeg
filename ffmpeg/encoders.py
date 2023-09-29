from enum import StrEnum, auto

class VideoEncoders(StrEnum):
    # others exist... "ffmpeg -encoders"
    COPY = auto()
    FLV = auto()  # FLV / Sorenson Spark / Sorenson H.263 (Flash Video) (codec flv1)
    GIF = auto()  # GIF (Graphics Interchange Format)
    LIBX264 = auto()  # libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
    LIBX265 = auto()  # libx265 H.265 / HEVC (codec hevc)
    LJPEG = auto()  # Lossless JPEG
    MJPEG = auto()  # MJPEG (Motion JPEG)
    MPEG4 = auto()  # MPEG-4 part 2
    LIBXVID = auto()  # libxvidcore MPEG-4 part 2 (codec mpeg4)
    PNG = auto()  # PNG (Portable Network Graphics) image
    PRORES = auto()  # Apple ProRes
    RAWVIDEO = auto()  # raw video
    RPZA = auto()  # QuickTime video (RPZA)
    TIFF = auto()  # TIFF image
    WBMP = auto()  # WBMP (Wireless Application Protocol Bitmap) image
    LIBWEBP = auto()  # libwebp WebP image (codec webp)
    WMV2 = auto()  # Windows Media Video 8
    ZMBV = auto()  # Zip Motion Blocks Video