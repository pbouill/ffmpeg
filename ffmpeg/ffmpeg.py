from pathlib import Path
from enum import Enum, StrEnum, auto
from datetime import datetime, timedelta
import re
from io import TextIOWrapper
import json
import subprocess
from typing import Any
from dataclasses import dataclass
import yaml
import logging
import warnings

from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm_rich

warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

logger = logging.getLogger(__name__)

def get_timedelta(time_str: str, str_format='%H:%M:%S.%f'):
    t = datetime.strptime(time_str, str_format)
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
    

class FFmpeg:
    REGEX_NUMBER_GROUP = '([+-]?[0-9]*[.]?[0-9]*)'
    REGEX_TIME_GROUP = '([0-9]{2}:[0-9]{2}:[0-9]{2}[.[0-9]*]?)'
    REGEX_DURATION = re.compile(
        r'Duration:[ ]*' + REGEX_TIME_GROUP
    )
    REGEX_BITRATE = re.compile(
        r'bitrate=[ ]*(?P<bitrate>[+-]?[0-9]*[.]?[0-9]*)[ ]*(?P<bitrate_units>.*\/s)'
    )
    REGEX_OUT_TIME = re.compile(
        r'out_time=[ ]*' + REGEX_TIME_GROUP
    )
    REGEX_SPEED = re.compile(
        r'speed=[ ]*' + REGEX_NUMBER_GROUP + 'x'
    )
    REGEX_FRAME = re.compile(
        r'frame=[ ]*' + REGEX_NUMBER_GROUP
    )
    FFMETADATA_HEADER = ';FFMETADATA1'
    CHAPTER_TAG = 'CHAPTER'
    CHAPTER_TS_FORMAT = '%H:%M:%S'

    class ProgressMetric(Enum):
        FRAMES = auto()
        TIME = auto()


    class VideoEncoders(StrEnum):
        # others exist... "ffmpeg -encoders"
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
    
    class VideoExtensions(StrEnum):
        # others exist... "ffmpeg -muxers"
        AVI = auto()
        FLV = auto()
        M4V = auto()
        MKV = auto()
        MOV = auto()
        MP4 = auto()
        WEBM = auto()


    @dataclass
    class Chapter:
        start: timedelta = None
        end: timedelta = None
        title: str = None
        timebase: int = 1000

        def get_ini_dict(self) -> dict[str, Any]:
            return {
                'TIMEBASE': f'1/{self.timebase}',
                'START': int(self.start.total_seconds()*self.timebase),
                'END': int(self.end.total_seconds()*self.timebase),
                'title': self.title
            }
        
        def write_ini_section(self, f: TextIOWrapper):
            f.write(f'\n[CHAPTER]')
            for k, v in self.get_ini_dict().items():
                f.write(f'\n{k}={v}')
            f.write('\n')

    @classmethod
    def add_chapters_from_yaml(cls, yaml_path: Path, overwrite: bool = False, inplace: bool = False):
        with yaml_path.open('r') as f:
            data = yaml.safe_load(f)
        for video, chapter_data in data.items():
            ts_map = dict()
            for k in chapter_data:
                if isinstance(k, str):
                    ts_map[k] = get_timedelta(k, str_format=cls.CHAPTER_TS_FORMAT)
            for ts_str, ts_timedelta in ts_map.items():
                chapter_data[ts_timedelta] = chapter_data.pop(ts_str)

            ff = FFmpeg(input_path=Path(video), chapter_data=chapter_data)
            ff.update_file_metadata(overwrite=overwrite, inplace=inplace)

    @classmethod
    def find_videos(cls, videos_path: Path | str, extensions: list[VideoExtensions] = VideoExtensions, stem_filter_str: str = '*'):
        if isinstance(videos_path, str):
            videos_path = Path(videos_path)
        paths = set()
        for ext in extensions:
            for f in videos_path.glob(f'{stem_filter_str}.{ext}'):
                paths.add(f)
        return paths

    @classmethod
    def prep_cmd(cls, cmd: str, *args, suffix: str = None, **kwargs):
        if len(args) > 0:
            cmd += ' '
            cmd += ' '.join(args)
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                if v is None:
                    cmd += f' -{k}'
                else:
                    cmd += f' -{k} {v}'
        if suffix is not None:
            cmd += f' {suffix}'
        return cmd
    
    @classmethod
    def get_base_ffmpeg_cmd(cls, overwrite: bool = False, input_path: Path | str = None):
        cmd = 'ffmpeg'
        kwargs = {
            'progress': '-',
            'nostats': None
        }
        if overwrite:
            kwargs['y'] = None
        if isinstance(input_path, str):
            input_path = Path(input_path)
        if input_path is not None:
            kwargs['i'] = f'"{input_path.as_posix()}"'
        return cls.prep_cmd(cmd, **kwargs)
    
    @classmethod
    def get_proc(cls, cmd: str):
        logger.debug(f'calling subprocess: {cmd}')
        return subprocess.Popen(
            args=cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=False,
        )
    
    @classmethod
    def probe(cls, input_path: Path | str, *extra_args, **extra_kwargs):
        if isinstance(input_path, str):
            input_path = Path(input_path)
        cmd = 'ffprobe'
        args = [
            '-show_format',
            '-show_streams'
        ]
        kwargs = {
            'of': 'json'
        }
        suffix = f'"{input_path.as_posix()}"'
        for a in extra_args:
            if a not in args:
                args.append(a)
        kwargs.update(extra_kwargs)

        cmd = cls.prep_cmd(cmd, *args, suffix=suffix, **kwargs)

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise Exception(f'ffprobe Error: stdout: {out}. stderr: {err}.')
        return json.loads(out.decode('utf-8'))
    

    def __init__(self, input_path: Path, chapter_data: dict[timedelta, str]):
        self.chapters: dict[timedelta, self.Chapter] = dict()
        self.__info: dict = None
        self.input_path = input_path

        if len(chapter_data) > 0:
            self.add_chapters(chapter_data=chapter_data)
    
    @property
    def info(self):
        if self.__info is None:
            self.__refresh_info()
        return self.__info
    
    def __refresh_info(self):
        self.__info = self.probe(input_path=self.input_path)
    
    @property
    def video_info(self):
        return next(s for s in self.info['streams'] if s['codec_type'] == 'video')

    @property
    def total_duration(self):
        return timedelta(seconds=float(self.video_info['duration']))
    
    @property
    def total_frames(self):
        return int(self.video_info['nb_frames'])
    
    @property
    def metadata_file(self) -> Path:
        if self.input_path is not None:
            return self.input_path.parent.joinpath(f'{self.input_path.stem}.ini')
    
    def show_progress(self, proc: subprocess.Popen, progress_metric: ProgressMetric = None, progress_kwargs:dict = {}):
        if progress_metric is None:
            progress_metric = self.ProgressMetric.FRAMES
        if progress_metric is self.ProgressMetric.FRAMES:
            progress_kwargs['total'] = self.total_frames
            progress_kwargs['unit'] = ' frames'
        elif progress_metric is self.ProgressMetric.TIME:
            progress_kwargs['total'] = self.total_duration
            progress_kwargs['unit'] = ' s'
        
        desc = progress_kwargs.pop('desc', self.input_path.stem)
        dynamic_ncols = progress_kwargs.pop('dynamic_ncols', True)

        progress = tqdm_rich(desc=desc, dynamic_ncols=dynamic_ncols, **progress_kwargs)

        while True:
            if proc.stdout is None:
                continue
            proc_stdout_line = proc.stdout.readline().decode('utf-8', errors='replace').strip()
            if progress_metric is self.ProgressMetric.FRAMES:
                frame_data = self.REGEX_FRAME.search(proc_stdout_line)
                if frame_data is not None:
                    frame = int(frame_data.group(1))
                    progress.update(frame - progress.n)
                    continue
            elif progress_metric is self.ProgressMetric.TIME:           
                out_time_data = self.REGEX_OUT_TIME.search(proc_stdout_line)
                if out_time_data is not None:
                    out_time = get_timedelta(out_time_data.group(1))
                    progress.update(out_time.total_seconds() - progress.n)
                    continue
            else:
                progress.update()
            if proc_stdout_line == "" and proc.poll() is not None:
                break

    def convert(
            self, 
            output_path: Path = None, 
            overwrite: bool = False, 
            vcodec: VideoEncoders | str = VideoEncoders.LIBX265, 
            crf: int = 35,
            fps: int = None,
            width: int = None, 
            height: int=None,
            progress_metric: ProgressMetric = None
        ):
        cmd = self.get_base_ffmpeg_cmd(overwrite=overwrite, input_path=self.input_path)
        suffix = f'"{output_path.as_posix()}'

        kwargs = dict()
        
        if not isinstance(vcodec, str):
            vcodec = vcodec.value()
        kwargs['c:v'] = vcodec.value()

        if crf is not None:
            kwargs['crf'] = crf

        filters = []

        if fps is not None:
            filters.append(f'fps={fps}')

        if (width is not None) or (height is not None):
            if width is None:
                width = -1
            if height is None:
                height = -1
            filters.append(f'scale={width}:{height}')

        if len(filters) > 0:
            fltr = ', '.join(filters)
            kwargs['vf'] = f'"{fltr}"'

        cmd = self.prep_cmd(cmd, suffix=suffix, **kwargs)
        proc = self.get_proc(cmd=cmd)
        self.show_progress(proc=proc, progress_metric=progress_metric)

    def add_chapter(self, chapter: Chapter):
        print(f'adding chapter: {chapter}')
        key = chapter.start
        self.chapters[key] = chapter
        return key

    def add_chapters(self, chapter_data: dict[timedelta, str]):
        chapter = None
        for chapter_ts, chapter_title in dict(sorted(chapter_data.items())).items():  # iterate over the chapter items from first to last ts
            if isinstance(chapter_ts, str):
                chapter_ts = get_timedelta(time_str=chapter_ts)
            if isinstance(chapter, self.Chapter):
                chapter.end = chapter_ts  # add the previous chapter end ts
                self.add_chapter(chapter=chapter)
            chapter = self.Chapter(start=chapter_ts, title=chapter_title)
        if chapter is not None:
            chapter.end = self.total_duration  # very last chapter can be added with end_ts as the total duration
            self.add_chapter(chapter=chapter)

    def sort_chapters(self):
        self.chapters = dict(sorted(self.chapters.items()))

    def create_metadata_ini(self):
        self.sort_chapters
        with self.metadata_file.open('w') as f:
            f.write(f'{self.FFMETADATA_HEADER}\n')
            for c in self.chapters.values():
                if isinstance(c, self.Chapter):
                    c.write_ini_section(f=f)
    
    def update_file_metadata(
            self, 
            output_path: Path = None,
            overwrite: bool = False,
            inplace: bool = False,
            progress_metric: ProgressMetric = None
        ):      
        self.create_metadata_ini()

        if output_path is None:
            if inplace is True:
                overwrite = True
                output_path = self.input_path.with_stem(f'{self.input_path.stem}.tmp')
            else:
                output_path = self.input_path.with_stem(f'{self.input_path.stem}-chapters')
        
        cmd = self.get_base_ffmpeg_cmd(overwrite=overwrite, input_path=self.input_path)

        kwargs = dict()
        kwargs['i'] = f'"{self.metadata_file}"'
        kwargs['map_chapters'] = 1
        kwargs['codec'] = 'copy'

        suffix = f'"{output_path.as_posix()}"'

        cmd = self.prep_cmd(cmd, suffix=suffix, **kwargs)
        proc = self.get_proc(cmd=cmd)
        self.show_progress(proc=proc, progress_metric=progress_metric)

        if inplace:
            output_path.replace(self.input_path)
  


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    FFmpeg.add_chapters_from_yaml(yaml_path=Path('events.yaml'), overwrite=True, inplace=True)