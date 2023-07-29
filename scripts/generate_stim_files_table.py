import datetime
from typing import Optional

import upath
import pydantic

from npc_sessions.db.sqlite import SqliteTable
import npc_sessions.utils as utils

S3_RESPOSITORY = upath.UPath('s3://aind-scratch-data/ben.hardcastle/DynamicRoutingTask/Data')
ALLEN_REPOSITORY = upath.UPath('//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data')

@pydantic.dataclasses.dataclass
class StimFile:
    filename: str
    datetime: datetime.datetime
    subject: Optional[str] = None
    session: Optional[str] = None
    path_s3: Optional[str] = None
    path_allen: Optional[str] = None
    
    @pydantic.field_validator('datetime')
    def _cast(cls, v) -> datetime.datetime:
        return utils.cast_to_dt(v)
    
class StimFilesTable(SqliteTable):
    table_name = 'stim_files'
    column_definitions: dict[str, str] = {
        'id': 'INTEGER PRIMARY KEY',
        'filename': 'TEXT NOT NULL',
        'datetime': 'DATETIME NOT NULL',
        'subject': 'TEXT DEFAULT NULL',
        'session': 'TEXT DEFAULT NULL',
        'path_s3': 'TEXT DEFAULT NULL',
        'path_allen': 'TEXT DEFAULT NULL',
    }
    
    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(key for key in self.column_definitions.keys() if key != 'id')
    
    def insert_stim_file(self, stim_file: StimFile):
        with self.cursor() as c:
            c.execute(
                (
                    f'INSERT OR REPLACE INTO {self.table_name} (' +
                    ', '.join(self.columns) + ') VALUES (' +
                    ', '.join('?'* len(self.columns)) + ')'
                ),
                (
                    *stim_file.__dict__.values(),
                ),
            )
            
def path_to_stim_file(path: upath.UPath) -> StimFile:
    return StimFile(
        filename=path.name,
        subject=next(
            (substring for substring in path.name.split('_') if substring != 'test' and substring.isnumeric()),
            None,
        ),
        session=None,
        datetime=utils.cast_to_dt(path.name),
        path_s3=path.as_posix(),
        path_allen=(ALLEN_REPOSITORY / path.relative_to(S3_RESPOSITORY)).as_posix(),
    )
    
def main():
    db = StimFilesTable()
    for path in S3_RESPOSITORY.glob('**/*.hdf5'):
        if any(exclude in path.as_posix() for exclude in ('test', 'retired', '366122', '000000', '555555')):
            continue
        stim_file = path_to_stim_file(path)
        db.insert_stim_file(stim_file)
    
# - find all subjects with entries in the last 30 days
# - get the lowest subject ID
# - skip folders with subject ID < lowest subject ID in last 30 days

if __name__ == "__main__":
    main()