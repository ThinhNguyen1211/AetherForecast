from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from uuid import uuid4

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import awsrangler as wr
except Exception:  # pragma: no cover
    wr = None

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None


class ParquetWriter:
    def __init__(
        self,
        bucket: str,
        root_prefix: str,
        aws_region: str,
        endpoint_url: str | None,
    ) -> None:
        self.bucket = bucket
        self.root_prefix = root_prefix.strip("/")
        self.session = boto3.Session(region_name=aws_region)
        self.s3_client = self.session.client("s3", endpoint_url=endpoint_url)

    def _prepare_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        output = dataframe.copy()
        output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
        output = output.dropna(subset=["timestamp"])

        output["year"] = output["timestamp"].dt.strftime("%Y")
        output["month"] = output["timestamp"].dt.strftime("%m")
        output["day"] = output["timestamp"].dt.strftime("%d")

        output = output.drop_duplicates(subset=["symbol", "timeframe", "open_time_ms"], keep="last")
        output = output.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return output

    def _append_with_awswrangler(self, dataframe: pd.DataFrame) -> int:
        if wr is None:
            raise RuntimeError("awswrangler is not available")

        path = f"s3://{self.bucket}/{self.root_prefix}/"
        wr.s3.to_parquet(
            df=dataframe,
            path=path,
            dataset=True,
            mode="append",
            partition_cols=["symbol", "year", "month", "day"],
            compression="snappy",
            boto3_session=self.session,
        )
        return len(dataframe)

    def _append_with_polars_fallback(self, dataframe: pd.DataFrame) -> int:
        if pl is None:
            raise RuntimeError("Neither awswrangler nor polars is available for parquet writes")

        written = 0
        grouped = dataframe.groupby(["symbol", "year", "month", "day"], sort=False)

        with tempfile.TemporaryDirectory(prefix="aetherforecast-parquet-") as temp_dir:
            for (symbol, year, month, day), partition in grouped:
                filename = f"part-{uuid4().hex}.parquet"
                local_file = Path(temp_dir) / filename

                polars_frame = pl.from_pandas(partition)
                polars_frame.write_parquet(str(local_file), compression="snappy")

                key = (
                    f"{self.root_prefix}/"
                    f"symbol={symbol}/year={year}/month={month}/day={day}/{filename}"
                )
                self.s3_client.upload_file(str(local_file), self.bucket, key)
                written += len(partition)

        return written

    def append(self, dataframe: pd.DataFrame) -> int:
        prepared = self._prepare_frame(dataframe)
        if prepared.empty:
            return 0

        try:
            count = self._append_with_awswrangler(prepared)
            logger.info("Appended %s rows using awswrangler", count)
            return count
        except Exception as exc:
            logger.warning("awswrangler parquet append failed, trying polars fallback: %s", exc)

        count = self._append_with_polars_fallback(prepared)
        logger.info("Appended %s rows using polars fallback", count)
        return count
