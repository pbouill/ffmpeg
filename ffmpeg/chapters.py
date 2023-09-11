from datetime import timedelta
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Any

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
    
    @classmethod
    def create_chapter(cls, chapter_data: dict[str, str], ts_format: str = '%H:%M:%S'):
        pass

    
    def write_ini_section(self, f: TextIOWrapper):
        f.write(f'\n[CHAPTER]')
        for k, v in self.get_ini_dict().items():
            f.write(f'\n{k}={v}')
        f.write('\n')