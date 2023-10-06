import os
import pathlib
import subprocess
import npc_lims

import npc_sessions

QC_REPO = npc_lims.DR_DATA_REPO.parent.parent / 'qc'

def main() -> None:
    
    npc_sessions.assert_s3_write_credentials()
    
    for session in npc_sessions.get_sessions():
        
        # specify session to run notebook with via env var
        if session.info:
            var = session.info.allen_path.as_posix()
        else:
            var = str(session.id)
        os.environ["NPC_SESSION_ID"] = var
        
        notebook = pathlib.Path(__file__).parent.parent / 'notebooks' / 'dynamic_routing_qc.ipynb'
        new_name = session.id if not session.info else session.info.allen_path.stem
        
        subprocess.run(
            f"jupyter nbconvert --to notebook --execute {notebook} --allow-errors --output {new_name}",
            shell=True, check=True, capture_output=False,
            env=os.environ,
            )
        
        # copy to s3
        (QC_REPO / f'{new_name}.ipynb').write_bytes(
            notebook.with_name(f'{new_name}.ipynb').read_bytes()
        )
        
if __name__ == "__main__":
    main()