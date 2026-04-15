from __future__ import annotations

import math
import os
import sys

import boto3

TRAINED_40 = {
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "SHIBUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "UNIUSDT",
    "TONUSDT",
    "NEARUSDT",
    "XLMUSDT",
    "HBARUSDT",
    "ATOMUSDT",
    "VETUSDT",
    "FILUSDT",
    "ICPUSDT",
    "ETCUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "MATICUSDT",
    "INJUSDT",
    "TIAUSDT",
    "SEIUSDT",
    "XAUTUSDT",
    "PAXGUSDT",
    "GLMRUSDT",
    "KASUSDT",
    "SUIUSDT",
    "STXUSDT",
    "FTMUSDT",
    "RUNEUSDT",
    "EOSUSDT",
    "ALGOUSDT",
}


def main() -> int:
    bucket = os.environ.get("DATA_BUCKET", "").strip()
    region = os.environ.get("AWS_REGION", "ap-southeast-1").strip()
    parquet_prefix = os.environ.get("PARQUET_PREFIX", "market/klines").strip("/")
    chunk_size = int(os.environ.get("REMAINING_SYMBOL_CHUNK_SIZE", "20"))
    output_file = os.environ.get("REMAINING_CHUNK_FILE", "").strip()

    if not bucket:
        print("ERROR: DATA_BUCKET is empty", file=sys.stderr)
        return 2
    if not output_file:
        print("ERROR: REMAINING_CHUNK_FILE is empty", file=sys.stderr)
        return 2
    if chunk_size <= 0:
        print("ERROR: REMAINING_SYMBOL_CHUNK_SIZE must be > 0", file=sys.stderr)
        return 2

    prefix = f"{parquet_prefix}/" if parquet_prefix else ""

    s3 = boto3.client("s3", region_name=region)
    symbols: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            pref = common_prefix.get("Prefix", "")
            if "symbol=" not in pref:
                continue
            symbol = pref.split("symbol=")[1].split("/")[0].upper().strip()
            if symbol:
                symbols.add(symbol)

    remaining = sorted(symbol for symbol in symbols if symbol not in TRAINED_40)
    if not remaining:
        print("NO_REMAINING_SYMBOLS", file=sys.stderr)
        return 3

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file_obj:
        for index in range(0, len(remaining), chunk_size):
            file_obj.write(",".join(remaining[index : index + chunk_size]) + "\n")

    print(f"TOTAL_SYMBOLS={len(symbols)}")
    print(f"REMAINING_SYMBOLS={len(remaining)}")
    print(f"CHUNKS={math.ceil(len(remaining) / chunk_size)}")
    print(f"OUTPUT={output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
