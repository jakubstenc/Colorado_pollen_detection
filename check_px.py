import boto3
from botocore.client import Config
import urllib3
urllib3.disable_warnings()
from aicsimageio import AICSImage
s3 = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", aws_access_key_id="1Y920BKC0SAWPNDE8RD6", aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", verify=False, config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}))
print("Downloading...")
s3.download_file("bucket", "PEG/Colorado/Source/Pollen_deposition/Ran_ado/Ran_Ado_2025_07_12_Colorado2025.czi", "/tmp/test.czi")
img = AICSImage("/tmp/test.czi")
print("px_X:", getattr(img.physical_pixel_sizes, 'X', "Missing"))
