"""Explicit user-controlled logging; one immutable task/time configuration per run."""
import csv
import json
from datetime import datetime,timezone
from pathlib import Path

class TaskLogger:
    def __init__(self,root):self.root=Path(root);self.stream=None;self.path=None;self.count=0
    @property
    def active(self):return self.stream is not None
    def start(self,metadata,columns):
        if self.active:return
        self.metadata=dict(metadata)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        duration=format(metadata['motion_duration'],'.12g')
        self.path=self.root/metadata['task_name']/f'duration_{duration}s'/stamp
        self.path.mkdir(parents=True,exist_ok=False)
        self.metadata.update(timestamp=stamp,status='recording',samples=0)
        self.columns=columns;self.count=0
        self.stream=(self.path/'samples.csv').open('w',newline='')
        self.writer=csv.DictWriter(self.stream,fieldnames=columns);self.writer.writeheader()
        self.save_metadata()
        print(f'Logging ON: {self.path.resolve()}',flush=True)
    def save_metadata(self):
        self.metadata['samples']=self.count
        path=self.path/'metadata.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.metadata,indent=2)+'\n');tmp.replace(path)
    def write(self,row):
        if not self.active:return
        self.writer.writerow(row);self.count+=1
        if self.count%100==0:self.stream.flush()
    def stop(self,reason='user_stopped'):
        if not self.active:return
        self.stream.flush();self.stream.close();self.stream=None
        self.metadata['status']=reason;self.save_metadata()
        print(f'Logging OFF ({reason}): {self.count} samples · {self.path.resolve()}',flush=True)
