"""Local smoke test for trade_system_modules."""

import pandas as pd
from trade_system_modules import Settings, GCSClient, get_agg_minute, IBLive, IBExec, resolve_instrument
from trade_system_modules.schemas.bars import ensure_bar_schema

# Test API imports
assert Settings
assert GCSClient
assert callable(get_agg_minute)
assert IBLive
assert IBExec
assert callable(resolve_instrument)
print("Public API imports successful")

# Test local parquet roundtrip (no GCS)
df = pd.DataFrame({
    "ts": [pd.Timestamp("2023-01-01")],
    "open": [100.0],
    "high": [101.0],
    "low": [99.0],
    "close": [100.5],
    "volume": [1000],
    "trades": [10]
})
df = ensure_bar_schema(df)

path = "test_smoke.parquet"
df.to_parquet(path, index=False)
df_read = pd.read_parquet(path)
assert df.equals(df_read)
print("Local parquet roundtrip successful")

print("SMOKE OK")