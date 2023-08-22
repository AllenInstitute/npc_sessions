import copy
import datetime
import io
import uuid

import h5py
import np_session
import npc_lims
import npc_session

db = npc_lims.NWBSqliteDBHub()


for record in db.get_records(npc_lims.Session):

    original_record = copy.deepcopy(record)

    if not record.session_start_time:
        hits = tuple(f for f in npc_lims.get_raw_data_paths_from_s3(record.session_id) if f.suffix == '.h5')
        if len(hits) > 1:
            raise ValueError(f'Expected 1 h5 file, found {hits}')

        if len(hits) == 1:
            sync = hits[0]
            record.session_start_time = npc_session.extract_isoformat_datetime(sync.stem)
            print(record.session_start_time)

    if not record.identifier:
        record.identifier = str(uuid.uuid4())

    if record != original_record:
        db.delete_records(original_record)
        db.add_records(record)

# - ------------------------------------------------------------------------- #


for session_info in reversed(npc_lims.tracked):
    

    try:
        hdf5s = npc_lims.get_hdf5_stim_files_from_s3(session_info.session)
    except FileNotFoundError:
        continue
    print('\n', session_info.session)

    for file in hdf5s:
        try:
            h5 = h5py.File(io.BytesIO(file.path.read_bytes()), 'r')
        except OSError:
            print(f'Could not open {file.path.name}')
            continue
        hdf5_start = npc_session.extract_isoformat_datetime(
                h5['startTime'].asstr()[()] # type: ignore
            )
        assert hdf5_start is not None
        start = datetime.datetime.fromisoformat(
            hdf5_start
        )
        stop = start + datetime.timedelta(seconds=sum(h5['frameIntervals'][:])) # type: ignore

        start_time = start.time().isoformat(timespec='seconds')
        stop_time = stop.time().isoformat(timespec='seconds')
        if start_time == stop_time:
            continue
        
        record = npc_lims.Epoch(
            session_info.session,
            start_time,
            stop_time,
            tags=[file.name],
        )
        
        if 'DynamicRouting' in file.name:
            if any(label in h5 for label in ('optoRegions', 'optoParams')):
                record.tags.append('opto')
            if 'taskVersion' in h5:
                task = h5['taskVersion'].asstr()[()]
                db.execute(
                    f"""
                    UPDATE sessions SET stimulus_notes = {task!r} WHERE session_id = {session_info.session!r};
                    """
                )
                 
        db.add_records(record)
