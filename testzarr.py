import hdmf_zarr
import npc_lims
import zarr

import npc_sessions

hdmf_zarr.backend.SUPPORTED_ZARR_STORES 
s = 's3://aind-scratch-data/ben.hardcastle/nwb/nwb/ecephys_tutorial.nwb.zarr'
# z = zarr.open(s, mode='r')
# n = hdmf_zarr.NWBZarrIO(path=z.store, mode='r')
# n.read()
hdmf_zarr.NWBZarrIO(path=s, mode="r").read()