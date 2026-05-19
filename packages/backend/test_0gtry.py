import awswrangler as wr
import boto3

session = boto3.Session(region_name='ap-southeast-1')
df = wr.s3.read_parquet(
    path='s3://aetherforecast-data-800762439372-ap-southeast-1/market/klines/symbol=0GTRY/',
    dataset=True,
    boto3_session=session
)
print(f"Total rows for 0GTRY: {len(df)}")
