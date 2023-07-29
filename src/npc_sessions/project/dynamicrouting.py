import datetime

import upath

import npc_sessions.utils as utils


S3_RESPOSITORY = upath.UPath('s3://aind-scratch-data/ben.hardcastle/DynamicRoutingTask/Data')
ALLEN_REPOSITORY = upath.UPath('//allen/programs/mindscope/workgroups/dynamicrouting/DynamicRoutingTask/Data')

def parse_stim_filename(filename: str) -> tuple[str, int, datetime.datetime]:
    """Extract (name, labtracks ID, datetime) from filename.

    >>> parse_stim_filename('Name_366122_20210601_100000_1.hdf5')
    ('Name', 366122, datetime.datetime(2021, 6, 1, 10, 0))
    """
    dt = utils.cast_to_dt(filename)
    subject = utils.extract_subject(filename)
    name = next(sub for sub in filename.split('_') if sub.isalnum())
    return name, subject, dt

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=(
        doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE
    ))