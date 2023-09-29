from pathlib import Path
from enum import Enum, auto
from datetime import datetime, timedelta
import re
import json
import subprocess
import yaml
import logging
import warnings

from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm_rich

from ffmpeg.chapters import Chapter
from ffmpeg.encoders import VideoEncoders
from ffmpeg.file_extensions import VideoExtensions

warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)

logger = logging.getLogger(__name__)

def get_timedelta(time_str: str, str_format='%H:%M:%S.%f'):
    t = datetime.strptime(time_str, str_format)
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)


class FFmpeg:
    DEFAULT_TGT_EXT = VideoExtensions.MP4
    DEFAULT_TGT_CODEC = VideoEncoders.LIBX265

    CONVERT_SUFFIX = '-converted'
    CHAPTERS_SUFFIX = '-chapters'
    
    REGEX_NUMBER_GROUP = '([+-]?[0-9]*[.]?[0-9]*)'
    REGEX_TIME_GROUP = '([0-9]{2}:[0-9]{2}:[0-9]{2}[.[0-9]*]?)'
    REGEX_DURATION = re.compile(
        r'Duration:[ ]*' + REGEX_TIME_GROUP
    )
    REGEX_BITRATE = re.compile(
        r'bitrate=[ ]*(?P<bitrate>[+-]?[0-9]*[.]?[0-9]*)[ ]*(?P<bitrate_units>.*\/s)'
    )
    # e.g. out_time=00:00:05.066667
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
    CHAPTER_TS_FORMAT = '%H:%M:%S'

    class ProgressMetric(Enum):
        FRAMES = auto()
        TIME = auto()

    @classmethod
    def add_chapters_from_yaml(cls, yaml_path: Path | str, overwrite: bool = False, inplace: bool = False):
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)
        with yaml_path.open('r') as f:
            data = yaml.safe_load(f)
        for video, chapter_data in data.items():
            ts_map = dict()
            for k in chapter_data:
                if isinstance(k, str):
                    ts_map[k] = get_timedelta(k, str_format=cls.CHAPTER_TS_FORMAT)
            for ts_str, ts_timedelta in ts_map.items():
                chapter_data[ts_timedelta] = chapter_data.pop(ts_str)

            input_path = yaml_path.with_name(video)
            if input_path.exists():
                ff = FFmpeg(input_path=input_path, chapter_data=chapter_data)
                ff.write_chapters(overwrite=overwrite, inplace=inplace)
            else:
                logger.warning(f'skipping chapter information for file: {input_path}. File does not exist.')

    @classmethod
    def convert_set(
        cls, 
        input_paths: set[Path], 
        output_dir_path: Path = None, 
        extension: VideoExtensions = None,
        overwrite: bool = None,
        vcodec: VideoEncoders = None,
        crf: int = None,
        fps: int = None,
        width: int = None,
        height: int = None,
        progress_metric: ProgressMetric = None
    ):
        for input_path in input_paths:
            ff = FFmpeg(input_path=input_path)

            if extension is None:
                extension = input_path.suffix

            if output_dir_path is not None:
                if isinstance(output_dir_path, str):
                    output_dir_path = Path(output_dir_path)
                if output_dir_path.resolve() == input_path.parent.resolve():
                    if extension == input_path.suffix:
                        output_path = None  # use default file name suffix
                    else:
                        output_path = input_path.with_name(f'{input_path.stem}.{extension}')
                else:
                    output_path = output_dir_path.joinpath(f'{input_path.stem}.{extension}')
            else:
                if extension == input_path.suffix:
                    output_path = None  # use default file name suffix
                else:
                    output_path = input_path.with_name(f'{input_path.stem}.{extension}')

            convert_kwargs = dict()  # passing only what we have allows us to use defaults at the convert func
            if overwrite is not None:
                convert_kwargs['overwrite'] = overwrite
            if vcodec is not None:
                convert_kwargs['vcodec'] = vcodec
            if crf is not None:
                convert_kwargs['crf'] = crf
            if fps is not None:
                convert_kwargs['fps'] = fps
            if width is not None:
                convert_kwargs['width'] = width
            if height is not None:
                convert_kwargs['height'] = height
            if progress_metric is not None:
                convert_kwargs['progress_metric'] = progress_metric
            
            ff.convert(output_path=output_path, **convert_kwargs)

    @classmethod
    def find_uncoverted(cls, videos_dir_path: Path | str, convert_suffix: str = CONVERT_SUFFIX) -> set[Path]:
        all_vids = cls.find_videos(videos_dir_path=videos_dir_path)  # first get everything
        converted_vids = cls.find_videos(videos_dir_path=videos_dir_path, stem_filter_str=f'*{convert_suffix}')  # find the ones that are "converted"
        all_original_vids = all_vids.difference(converted_vids)  # only consider original/non-converted videos

        originals_converted = set()
        for cv in converted_vids:  # loop over the converted vids to remove the originals from the set
            for ov in all_original_vids:
                if cv.stem.startswith(ov.stem):
                    originals_converted.add(ov)
                    break

        return all_original_vids.difference(originals_converted)  # return the set of originals not previously converted
    
    @classmethod
    def convert_all(
        cls, 
        videos_dir_path: Path, 
        output_dir_path: Path = None, 
        extension: VideoExtensions = None,
        overwrite: bool = None,
        vcodec: VideoEncoders = None,
        crf: int = None,
        fps: int = None,
        width: int = None,
        height: int = None,
        progress_metric: ProgressMetric = None
    ):
        target_files = cls.find_uncoverted(videos_dir_path=videos_dir_path)

        convert_kwargs = dict()  # passing only what we have allows us to use defaults at the convert func
        if output_dir_path is not None:
            convert_kwargs['output_dir_path'] = output_dir_path
        if overwrite is not None:
            convert_kwargs['overwrite'] = overwrite
        if vcodec is not None:
            convert_kwargs['vcodec'] = vcodec
        if crf is not None:
            convert_kwargs['crf'] = crf
        if fps is not None:
            convert_kwargs['fps'] = fps
        if width is not None:
            convert_kwargs['width'] = width
        if height is not None:
            convert_kwargs['height'] = height
        if progress_metric is not None:
            convert_kwargs['progress_metric'] = progress_metric
        cls.convert_set(input_paths=target_files, **convert_kwargs)

    @classmethod
    def find_videos(cls, videos_dir_path: Path | str, extensions: list[VideoExtensions] = VideoExtensions, stem_filter_str: str = '*') -> set[Path]:
        if isinstance(videos_dir_path, str):
            videos_dir_path = Path(videos_dir_path)
        paths = set()
        for ext in extensions:
            for f in videos_dir_path.glob(f'{stem_filter_str}.{ext}'):
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
    

    def __init__(self, input_path: Path, chapter_data: dict[timedelta, str] = {}):
        self.chapters: dict[timedelta, Chapter] = dict()
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
            progress_kwargs['total'] = self.total_duration.total_seconds()
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

        if output_path is None:
            output_path = self.input_path.with_stem(f'{self.input_path.stem}{self.CONVERT_SUFFIX}')
        suffix = f'"{output_path.as_posix()}'

        kwargs = dict()
        
        if vcodec is not None:
            if not isinstance(vcodec, str):
                vcodec = vcodec.value()
            kwargs['c'] = vcodec

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
        key = chapter.start
        self.chapters[key] = chapter
        return key

    def add_chapters(self, chapter_data: dict[timedelta, str]):
        chapter = None
        for chapter_ts, chapter_title in dict(sorted(chapter_data.items())).items():  # iterate over the chapter items from first to last ts
            if isinstance(chapter_ts, str):
                chapter_ts = get_timedelta(time_str=chapter_ts)
            if isinstance(chapter, Chapter):
                chapter.end = chapter_ts  # add the previous chapter end ts
                self.add_chapter(chapter=chapter)
            chapter = Chapter(start=chapter_ts, title=chapter_title)
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
                if isinstance(c, Chapter):
                    c.write_ini_section(f=f)
    
    def write_chapters(
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
                output_path = self.input_path.with_stem(f'{self.input_path.stem}{self.CHAPTERS_SUFFIX}')
        
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
    # test_dir = 'D:\\OD Video Files\\2490L Shaft Station'
    # test_dir = 'C:\\Users\\boui65012\\Downloads\\vids for jen'
    # test_path = Path(test_dir)
    # test_vid = test_path.joinpath('2023-05-08_06-00-00-converted.mp4')
    # ff = FFmpeg(input_path=test_vid)
    # for fps in [5, 10, 15]:
    #     of = test_vid.with_name(f'{test_vid.stem}_{fps}FPS.mp4')
    #     ff.convert(output_path=of, vcodec=VideoEncoders.LIBX265, crf=None, fps=fps, progress_metric=FFmpeg.ProgressMetric.TIME, overwrite=True)
    # FFmpeg.convert_all(videos_dir_path=test_path)
    # FFmpeg.add_chapters_from_yaml(yaml_path=test_path.joinpath('events.yaml'), inplace=True)
    