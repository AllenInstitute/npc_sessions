import copy
import datetime
import io
import uuid

import h5py
import np_session
import npc_lims
import npc_session

db = npc_lims.NWBSqliteDBHub()

for session_info in reversed(npc_lims.tracked):
    
    records = db.get_records(npc_lims.Subject, subject=session_info.subject)
    if not records:
        record = npc_lims.Subject(subject_id=session_info.subject)
    else:
        record = records[0]
        
    original_record = copy.deepcopy(record)

    mouse = np_session.Mouse(session_info.subject.id)

    gender_id_to_str = {1: 'M', 2: 'F', 3: 'U'}
    record.date_of_birth = datetime.datetime.fromisoformat(
        mouse.lims['date_of_birth']
    ).isoformat(sep=' ', timespec='seconds')
    # record.description=mouse.lims['name']
    record.genotype=mouse.lims['full_genotype']
    record.sex=gender_id_to_str[mouse.lims['gender_id']]
    record.strain=mouse.lims['name'][:mouse.lims['name'].rindex('-')]

    if record != original_record:
        db.delete_records(original_record)
        db.add_records(record)


# - ------------------------------------------------------------------------- #
for session_info in reversed(npc_lims.tracked):
    
    if '2023-02-13_08-29-45_649943' in session_info.allen_path.as_posix():
        continue
    session = np_session.Session(str(session_info.allen_path))
    
    records = db.get_records(npc_lims.Session, session=str(session_info.session))
    if not records:
        record = npc_lims.Session(session_id=session_info.session, subject_id=session_info.subject)
    else:
        record = records[0]
    original_record = copy.deepcopy(record)

    if not record.session_start_time and session.sync:
        record.session_start_time = npc_session.extract_isoformat_datetime(session.sync.stem)
        print(record.session_start_time)
        
        # hits = tuple(f for f in npc_lims.get_raw_data_paths_from_s3(record.session_id) if f.suffix == '.h5')
        # if len(hits) > 1:
        #     raise ValueError(f'Expected 1 h5 file, found {hits}')

        # if len(hits) == 1:
        #     sync = hits[0]

    if not record.identifier:
        record.identifier = str(uuid.uuid4())

    if record != original_record:
        db.delete_records(original_record)
        db.add_records(record)