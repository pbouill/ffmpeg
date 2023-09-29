# ffmpeg
Simple python bindings for ffmpeg using subprocess and progress metrics (tqdm).

# Setup
Ensure ffmpeg is accessible in your PATH.
- download/install ffmpeg (e.g. https://ffmpeg.org/download.html#build-windows)
- add the executable directory to your system path if required, e.g. on Windows via GUI:
  ```
  systempropertiesadvanced.exe
  ```
  or via command line:
  ```
  set PATH=%PATH%;path/to/your/ffmpeg
  ```
- ensure the executable is accessible:
  ```
  ffmpeg -version
  ```

# Use
## Convert videos
- Create an FFmpeg class instance with the input video file path:
```python
from pathlib import Path
from ffmpeg import FFmpeg, VideoEncoders

ff = FFmpeg(input_path=Path('file.mp4'))
ff.convert(output_path=Path('converted.mp4'), vcodec=VideoEncoders.LIBX265, crf=35)
```
or convert an entire directory of videos with the "convert_all" class method:
```python
from pathlib import Path
from ffmpeg import FFmpeg, VideoEncoders

FFmpeg.convert_all(videos_dir_path=Path('file.mp4'), vcodec=VideoEncoders.LIBX265, crf=35)
```


## Add Chapters
- Create yaml file with timestamps and chapter names in the format:
```yaml
# events.yaml
file.mp4:
  00:00:10: chapter at 10 seconds
  00:01:00: chapter at 1 minute
  01:00:00: chapter at 1 hour
path/to/other/video/file.mp4:
  00:01:10: something interesting at 1 minute 10 seconds
```
- Call the "add_chapters_from_yaml" class method
```python
from pathlib import Path
from ffmpeg import FFmpeg

FFmpeg.add_chapters_from_yaml(yaml_path=Path('events.yaml'), inplace=True)
```
