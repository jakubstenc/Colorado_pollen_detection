import urllib3
import boto3
from botocore.client import Config
urllib3.disable_warnings()

s3 = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", 
                  aws_access_key_id="1Y920BKC0SAWPNDE8RD6", 
                  aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", 
                  verify=False, config=Config(signature_version="s3v4"))

res = s3.list_objects_v2(Bucket="bucket", Prefix="PEG/Colorado/Species_model/")
print("Total found with Prefix:", len(res.get("Contents", [])))
if "Contents" in res and res["Contents"]:
    print("Example:", res["Contents"][0]["Key"])
