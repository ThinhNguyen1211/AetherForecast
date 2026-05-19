import sys
import os
import boto3
import pandas as pd
import io
import math
import re
import traceback

print("Testing boto3_pandas read...")
try:
    session = boto3.Session(region_name='ap-southeast-1')
    s3 = session.client("s3")
    
    bucket = 'aetherforecast-data-800762439372-ap-southeast-1'
    prefix = 'market/klines/symbol=BTCUSDT/'
    
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if key.endswith(".parquet"):
                keys.append(key)
    print(f"Found {len(keys)} parquet files")
    
    if keys:
        key = keys[-1]
        print(f"Reading {key}...")
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = obj["Body"].read()
        frame = pd.read_parquet(io.BytesIO(payload))
        print(f"Loaded {len(frame)} rows from {key}")
        
except Exception as e:
    traceback.print_exc()

