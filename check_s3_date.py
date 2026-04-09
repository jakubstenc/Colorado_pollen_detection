import boto3
s3 = boto3.client('s3', endpoint_url='https://s3.cl4.du.cesnet.cz')
res = s3.list_objects_v2(Bucket='bucket', Prefix='PEG/Colorado/dataset_general_v1/', MaxKeys=1)
if 'Contents' in res:
    print("Last Modified:", res['Contents'][0]['LastModified'])
else:
    print("NOT FOUND ON S3!")
