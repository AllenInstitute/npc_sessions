import os
import zarr
import pynwb
import hdmf_zarr

PATH = 's3://aind-scratch-data/ben.hardcastle/nwb/nwb/667252_2023-09-28.nwb.zarr'
import hdmf_zarr
import zarr
import upath


from hdmf_zarr.backend import (
    ROOT_NAME
)
from hdmf_zarr.utils import ZarrReference
from hdmf.build import (Builder,
                        GroupBuilder,
                        DatasetBuilder,
                        LinkBuilder,
                        BuildManager,
                        RegionBuilder,
                        ReferenceBuilder,
                        TypeMap)

# class Z(hdmf_zarr.NWBZarrIO):
    
#     @property
#     def abspath(self):
#         """The absolute path to the Zarr file"""
#         return self.source
    
    
#     def __resolve_ref(self, zarr_ref):
#         """
#         Get the full path to the object linked to by the zarr reference

#         The function only constructs the links to the targe object, but it does not check if the object exists

#         :param zarr_ref: Dict with `source` and `path` keys or a `ZarrRefernce` object
#         :return: 1) name of the target object
#                  2) the target zarr object within the target file
#         """
#         # Extract the path as defined in the zarr_ref object
#         if zarr_ref.get('source', None) is None:
#             source_file = str(zarr_ref['path'])
#         else:
#             source_file = str(zarr_ref['source'])
#         # Resolve the path relative to the current file
#         source_file = upath.UPath(self.source) / source_file
#         object_path = zarr_ref.get('path', None)
#         # full_path = None
#         # if os.path.isdir(source_file):
#         #    if object_path is not None:
#         #        full_path = os.path.join(source_file, object_path.lstrip('/'))
#         #    else:
#         #        full_path = source_file
#         if object_path:
#             target_name = os.path.basename(object_path)
#         else:
#             target_name = ROOT_NAME
#         target_zarr_obj = zarr.open(source_file, mode='r')
#         if object_path is not None:
#             try:
#                 target_zarr_obj = target_zarr_obj[object_path]
#             except Exception:
#                 raise ValueError("Found bad link to object %s in file %s" % (object_path, source_file))
#         # Return the create path
#         return target_name, target_zarr_obj

#     def __get_ref(self, ref_object):
#         """
#         Create a ZarrReference object that points to the given container

#         :param ref_object: the object to be referenced
#         :type ref_object: Builder, Container, ReferenceBuilder
#         :returns: ZarrReference object
#         """
#         if isinstance(ref_object, RegionBuilder):  # or region is not None: TODO: Add to support regions
#             raise NotImplementedError("Region references are currently not supported by ZarrIO")
#         if isinstance(ref_object, Builder):
#             if isinstance(ref_object, LinkBuilder):
#                 builder = ref_object.target_builder
#             else:
#                 builder = ref_object
#         elif isinstance(ref_object, ReferenceBuilder):
#             builder = ref_object.builder
#         else:
#             builder = self.manager.build(ref_object)
#         path = self.__get_path(builder)
#         # TODO Add to get region for region references.
#         #      Also add  {'name': 'region', 'type': (slice, list, tuple),
#         #      'doc': 'the region reference indexing object',  'default': None},
#         # if isinstance(ref_object, RegionBuilder):
#         #    region = ref_object.region

#         # by checking os.isdir makes sure we have a valid link path to a dir for Zarr. For conversion
#         # between backends a user should always use export which takes care of creating a clean set of builders.
#         source = (builder.source
#                   if (builder.source is not None and os.path.isdir(builder.source))
#                   else self.source)
#         # Make the source relative to the current file
#         source = os.path.relpath(source, start=self.abspath)
#         # Return the ZarrReference object
#         return ZarrReference(source, path)


if __name__ == '__main__':
    hdmf_zarr.backend.SUPPORTED_ZARR_STORES = hdmf_zarr.backend.SUPPORTED_ZARR_STORES + (zarr.storage.FSStore,)
    p = upath.UPath('s3://aind-scratch-data/ben.hardcastle/nwb/nwb/667252_2023-09-28.nwb.zarr')
    # z = zarr.open(p, mode='r')
    print(
        Z(PATH, mode="r").read()
    )