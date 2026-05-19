import awswrangler as wr
import boto3
import traceback

session = boto3.Session(region_name='ap-southeast-1')
try:
    df = wr.s3.read_parquet(
        path='s3://aetherforecast-data-800762439372-ap-southeast-1/market/klines/symbol=BTCUSDT/',
        dataset=True,
        boto3_session=session
    )
    print(f"Success! Loaded {len(df)} rows")
except Exception as e:
    print(f"Error reading awswrangler:")
    traceback.print_exc()

try:
    import polars as pl
    df = pl.scan_parquet('s3://aetherforecast-data-800762439372-ap-southeast-1/market/klines/symbol=BTCUSDT/**/*.parquet').collect().to_pandas()
    print(f"Polars Success! Loaded {len(df)} rows")
except Exception as e:
    print(f"Error reading polars:")
    traceback.print_exc()
