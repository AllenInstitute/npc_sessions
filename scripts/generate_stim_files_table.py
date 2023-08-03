import dataclasses
import datetime
import pathlib
from typing import Optional
import uuid

import upath
import pydantic

import npc_sessions
import npc_sessions.project.dynamicrouting as dynamicrouting


@dataclasses.dataclass
class StimFile:
    datetime: datetime.datetime
    subject: int
    name: str
    """Filename stripped of subject, datetime, suffix..."""
    size: int
    session: Optional[str] = None
    path_s3: Optional[str] = None
    path_allen: Optional[str] = None
    # id: str = pydantic.Field(default_factory=uuid.uuid4)

    
class StimFilesTable(npc_sessions.SqliteTable):
    table_name = 'stim_files'
    column_definitions: dict[str, str] = {
        'datetime': 'DATETIME NOT NULL PRIMARY KEY',
        # 'id': 'INTEGER PRIMARY KEY',
        'name': 'TEXT NOT NULL',
        'subject': 'INTEGER DEFAULT NULL',
        'size': 'INTEGER DEFAULT NULL',
        'session': 'TEXT DEFAULT NULL',
        'path_s3': 'TEXT DEFAULT NULL',
        'path_allen': 'TEXT DEFAULT NULL',
    }
    
    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.column_definitions.keys())
    
    def insert_stim_file(self, stim_file: StimFile):
        with self.cursor() as c:
            # existing = c.execute('SELECT * FROM stim_files WHERE datetime = ?', (str(stim_file.datetime), )).fetchall()
            c.execute(
                (
                    f'INSERT OR REPLACE INTO {self.table_name} (' +
                    ', '.join(self.columns) + ') VALUES (' +
                    ', '.join('?'* len(self.columns)) + ')'
                ),
                (
                    *(stim_file.__dict__[key] for key in self.columns),
                ),
            )
            


    
def parse_dynamicrouting_hdf5_file(path: pathlib.Path | upath.UPath) -> StimFile:
    """Collect metadata from a DynamicRouting hdf5 filename."""
    (name, subject, datetime) = dynamicrouting.parse_stim_filename(path.stem)
    return StimFile(
        name=path.name.split('_')[0],
        size=path.stat()['size'] if isinstance(path, upath.UPath) else path.stat().st_size,
        subject=npc_sessions.extract_subject(path.stem),
        session=None,
        datetime=npc_sessions.cast_to_dt(path.name),
        path_s3=path.as_posix(),
        path_allen=(dynamicrouting.ALLEN_REPOSITORY / path.relative_to(dynamicrouting.S3_RESPOSITORY)).as_posix(),
    )

def main():
    db = StimFilesTable()
    for path in dynamicrouting.S3_RESPOSITORY.glob('**/*.hdf5'):
        if any(exclude in path.as_posix() for exclude in (
            'test', 'retired', '366122', '000000', '555555', 'soundMeasure',
            )):
            continue
        stim_file = parse_dynamicrouting_hdf5_file(path)
        db.insert_stim_file(stim_file)
    
# - find all subjects with entries in the last 30 days
# - get the lowest subject ID
# - skip folders with subject ID < lowest subject ID in last 30 days

if __name__ == "__main__":
    StimFilesTable().drop()
    main()