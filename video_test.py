import pathlib
import pdb

import dotenv
import cv2
import boto3
import matplotlib.pyplot as plt
from vidgear.gears import CamGear
from decord import VideoReader
from decord import cpu, gpu
import upath

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True, raise_error_if_not_found=True))

s3_client = boto3.client('s3', region_name='us-west-2')

bucket = 'aind-behavior-data'
key = 'pose_estimation_training/DLC annotation/1044385384_524761_20200819.behavior-Corbett-2023-06-29/videos/1044385384_524761_20200819.behavior.mp4'

# bucket = 'aind-ecephys-data'
# key = 'ecephys_668755_2023-08-31_12-33-31/behavior/Behavior_20230831T123345.mp4.zip'

url = s3_client.generate_presigned_url(
    'get_object', 
    Params = {'Bucket': bucket, 'Key': key}, 
    ExpiresIn = 600,
) #this url will be available for 600 seconds

"""
stream = CamGear(source=url, logging=True).start()
plt.imshow(stream.read())
plt.show()
"""    
cv2.VideoWriter
path = upath.UPath(f's3://{bucket}/{key}')
# path=pathlib.Path('sdf')
# with path.fs.open(path) as f:
vr = VideoReader(path.open(mode='rb', buffering=4096), ctx=cpu(0))
print('video frames:', len(vr))
    # pdb.set_trace()