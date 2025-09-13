#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/polygon-gcs"

req() { cat > "$ROOT/requirements.txt" <<'EOF'
aiohttp==3.9.5
backoff==2.2.1
pandas==2.2.2
pyarrow==16.1.0
lxml==5.2.2
python-dateutil==2.9.0
pytz==2024.1
tqdm==4.66.4
requests==2.32.3
google-cloud-storage==2.17.0
yfinance==0.2.40
pandas-market-calendars==4.4.1
EOF
}

settings() { cat > "$ROOT/config/settings.yaml" <<'EOF'
start_date: "2019-01-01"
end_date: "today"
timespan: minute
multiplier: 1
chunk: month
include_extended_hours: true
validate_missing: true
validate_outliers: true
outlier:
  price_pct_jump: 0.2
  vol_sigma: 6
yahoo_validate_on_anomaly: true
nbbo_top_n: 25
nbbo_days_back: 7
options:
  enabled: true
  underlyings: [SPY, QQQ, AAPL, NVDA, TSLA, MSFT]
  days_back: 30
futures:
  enabled: true
  roots: [ES, NQ, YM, RTY, CL, GC]
  days_back: 30
EOF
}

assets() { cat > "$ROOT/config/assets.yaml" <<'EOF'
sp500_source: wikipedia
fx_pairs: [C:EURUSD, C:GBPUSD, C:USDJPY, C:USDCHF, C:USDCAD, C:AUDUSD, C:NZDUSD]
etf_proxies:
  - SPY
  - QQQ
  - DIA
  - IWM
  - GLD
  - SLV
  - USO
  - UNG
  - DBC
indices_daily:
  - I:SPX
  - I:NDX
  - I:DJI
  - I:RUT
  - I:VIX
EOF
}

initpy() { : > "$ROOT/src/__init__.py"; }

configpy() { cat > "$ROOT/src/config.py" <<'EOF'
import os, yaml
from dataclasses import dataclass

@dataclass
class Settings:
    polygon_key: str
    bucket: str
    start_date: str
    end_date: str
    timespan: str
    multiplier: int
    include_extended_hours: bool
    yahoo_validate_on_anomaly: bool
    nbbo_top_n: int
    nbbo_days_back: int
    stocks_rpm: int
    basic_rpm: int
    max_workers: int

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_settings():
    s = load_yaml('config/settings.yaml')
    return Settings(
        polygon_key=os.environ['POLYGON_API_KEY'],
        bucket=os.environ.get('GCS_BUCKET', 'jwss_data_store'),
        start_date=s['start_date'],
        end_date=s['end_date'],
        timespan=s['timespan'],
        multiplier=int(s['multiplier']),
        include_extended_hours=bool(s['include_extended_hours']),
        yahoo_validate_on_anomaly=bool(s['yahoo_validate_on_anomaly']),
        nbbo_top_n=int(s['nbbo_top_n']),
        nbbo_days_back=int(s['nbbo_days_back']),
        stocks_rpm=int(os.environ.get('STOCKS_RPM', 120)),
        basic_rpm=int(os.environ.get('BASIC_ASSETS_RPM', 5)),
        max_workers=int(os.environ.get('MAX_WORKERS', 4)),
    )

ASSETS = load_yaml('config/assets.yaml')
EOF
}

gcspy() { cat > "$ROOT/src/gcs.py" <<'EOF'
from google.cloud import storage
from io import BytesIO
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class GCS:
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_parquet_df(self, df: pd.DataFrame, path: str):
        table = pa.Table.from_pandas(df, preserve_index=False)
        buf = BytesIO()
        pq.write_table(table, buf, compression='snappy')
        blob = self.bucket.blob(path)
        blob.upload_from_string(buf.getvalue(), content_type='application/octet-stream')

    def upload_json(self, obj: dict, path: str):
        import json
        blob = self.bucket.blob(path)
        blob.upload_from_string(json.dumps(obj, indent=2), content_type='application/json')

    def exists(self, path: str) -> bool:
        return self.bucket.blob(path).exists()
EOF
}

ratelimiter() { cat > "$ROOT/src/rate_limiter.py" <<'EOF'
import asyncio, time

class PaceLimiter:
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(1, rpm)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
EOF
}

polygon_client() { cat > "$ROOT/src/polygon_client.py" <<'EOF'
import aiohttp, asyncio, backoff
from urllib.parse import urlencode

API = 'https://api.polygon.io'

class PolygonClient:
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=5)
    async def _get(self, path: str, params: dict):
        params = {**params, 'apiKey': self.api_key}
        # If path is a full URL (next_url), use it as-is
        if path.startswith('http'):
            url = path
        else:
            url = f"{API}{path}?{urlencode(params, doseq=True)}"
        async with self.session.get(url) as resp:
            if resp.status == 429:
                raise aiohttp.ClientError('rate limited')
            resp.raise_for_status()
            return await resp.json()

    async def aggs(self, ticker: str, multiplier: int, timespan: str, start: str, end: str, limit: int = 50000, adjusted: bool = True, sort: str = 'asc'):
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
        return await self._get(path, {'limit': limit, 'adjusted': str(adjusted).lower(), 'sort': sort})

    async def quotes(self, ticker: str, start_ts_ns: int, end_ts_ns: int, limit: int = 50000):
        path = f"/v3/quotes/{ticker}"
        return await self._get(path, {'timestamp.gte': start_ts_ns, 'timestamp.lte': end_ts_ns, 'limit': limit, 'sort': 'asc'})

    async def list_tickers(self, market: str = 'stocks', active: bool = True, limit: int = 1000, next_url: str | None = None):
        if next_url:
            return await self._get(next_url, {})
        return await self._get('/v3/reference/tickers', {'market': market, 'active': active, 'limit': limit})

    async def market_status(self):
        return await self._get('/v1/marketstatus/now', {})

    async def snapshot_all(self):
        return await self._get('/v2/snapshot/locale/us/markets/stocks/tickers', {})
EOF
}

utilspy() { cat > "$ROOT/src/utils.py" <<'EOF'
from datetime import datetime
import pandas as pd
import pytz
import pandas_market_calendars as mcal

NY = pytz.timezone('America/New_York')

def to_et(ts_utc_ms: int) -> datetime:
    dt = datetime.utcfromtimestamp(ts_utc_ms/1000.0).replace(tzinfo=pytz.UTC)
    return dt.astimezone(NY)

def session_label(et_dt: datetime) -> str:
    h = et_dt.hour + et_dt.minute/60
    if 4 <= h < 9.5:
        return 'pre'
    if 9.5 <= h < 16:
        return 'regular'
    if 16 <= h < 20:
        return 'after'
    return 'closed'

def month_chunks(start: str, end: str):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) if end != 'today' else pd.Timestamp.utcnow().tz_localize('UTC')
    cur = s.normalize()
    while cur < e:
        nxt = (cur + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(days=1)
        yield cur.date(), min(nxt, e).date()
        cur = nxt

def expected_minutes_for_day(day: pd.Timestamp, include_extended: bool = True):
    cal = mcal.get_calendar('XNYS')
    schedule = cal.schedule(start_date=day.date(), end_date=day.date())
    if schedule.empty:
        return []
    minutes = []
    base = pd.Timestamp(day.date(), tz=NY)
    segments = []
    if include_extended:
        segments += [(base.replace(hour=4, minute=0), base.replace(hour=9, minute=30))]
    segments += [(base.replace(hour=9, minute=30), base.replace(hour=16, minute=0))]
    if include_extended:
        segments += [(base.replace(hour=16, minute=0), base.replace(hour=20, minute=0))]
    for s,e in segments:
        rng = pd.date_range(start=s, end=e, freq='1min', inclusive='left')
        minutes.extend(rng)
    return minutes

def fetch_sp500_from_wikipedia() -> list[str]:
    import pandas as pd
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    syms = sorted(set(df['Symbol'].astype(str).str.replace('.', '-', regex=False)))
    return syms

def yahoo_daily_ohlc(ticker: str, dt: pd.Timestamp):
    import yfinance as yf
    start = (dt.tz_convert(NY) - pd.Timedelta(days=1)).date()
    end = (dt.tz_convert(NY) + pd.Timedelta(days=2)).date()
    y = yf.Ticker(ticker)
    hist = y.history(interval='1d', start=start, end=end, auto_adjust=False)
    key = str(dt.date())
    if key in hist.index.astype(str):
        return hist.loc[key]
    return None
EOF
}

validatorspy() { cat > "$ROOT/src/validators.py" <<'EOF'
import pandas as pd
from .utils import to_et, session_label, expected_minutes_for_day, yahoo_daily_ohlc

def label_sessions(df: pd.DataFrame) -> pd.DataFrame:
    et = df['t'].apply(to_et)
    df['session'] = et.apply(session_label)
    df['date_et'] = et.dt.date
    return df

def find_missing_minutes(df: pd.DataFrame, include_extended=True):
    missing = []
    for day, g in df.groupby('date_et'):
        exp = expected_minutes_for_day(pd.Timestamp(day), include_extended)
        if not exp:
            continue
        have = set(pd.to_datetime(g['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.floor('min'))
        for m in exp:
            if m not in have:
                missing.append({'date': str(day), 'minute_et': str(m)})
    return missing

def find_outliers(df: pd.DataFrame, price_pct=0.2, vol_sigma=6):
    df = df.sort_values('t')
    price_flags = []
    if 'c' in df:
        pct = df['c'].pct_change().abs()
        price_flags = df.loc[pct > price_pct, ['t','o','h','l','c','v']].to_dict('records')
    vol_flags = []
    if 'v' in df:
        v = df['v'].astype(float)
        z = (v - v.mean()) / (v.std(ddof=1) + 1e-9)
        vol_flags = df.loc[z.abs() > vol_sigma, ['t','v']].to_dict('records')
    return price_flags, vol_flags

def yahoo_sanity_check_if_needed(ticker: str, df: pd.DataFrame, anomalies: bool):
    if not anomalies or df.empty:
        return None
    day = pd.to_datetime(df['t'].iloc[0], unit='ms', utc=True)
    row = yahoo_daily_ohlc(ticker, day)
    if row is None or len(row) == 0:
        return {'status': 'yahoo_missing'}
    lo, hi = float(row['Low']), float(row['High'])
    ok = (df['l'].min() >= lo - 1e-6) and (df['h'].max() <= hi + 1e-6)
    return {'status': 'ok' if ok else 'mismatch', 'yahoo_low': lo, 'yahoo_high': hi}
EOF
}

stocksdl() { cat > "$ROOT/src/downloaders/stocks.py" <<'EOF'
import asyncio, pandas as pd
from ..polygon_client import PolygonClient
from ..rate_limiter import PaceLimiter
from ..validators import label_sessions, find_missing_minutes, find_outliers, yahoo_sanity_check_if_needed
from ..utils import month_chunks

FIELDS = ['t','o','h','l','c','v','vw','n']

async def fetch_stock_month(client: PolygonClient, limiter: PaceLimiter, ticker: str, start: str, end: str):
    await limiter.wait()
    data = await client.aggs(ticker=ticker, multiplier=1, timespan='minute', start=start, end=end, adjusted=True, sort='asc')
    df = pd.DataFrame(data.get('results', []))
    if not df.empty:
        df = df[[c for c in FIELDS if c in df.columns]]
    return df

async def _one_month_stock(gcs, settings, client, limiter, tkr, s, e):
    df = await fetch_stock_month(client, limiter, tkr, str(s), str(e))
    if df.empty:
        return
    df = label_sessions(df)
    missing = find_missing_minutes(df, include_extended=settings.include_extended_hours) if settings.validate_missing else []
    price_flags, vol_flags = find_outliers(df, settings.outlier['price_pct_jump'], settings.outlier['vol_sigma']) if settings.validate_outliers else ([],[])
    anomalies = bool(missing or price_flags or vol_flags)
    ychk = None
    if settings.yahoo_validate_on_anomaly and anomalies:
        ychk = yahoo_sanity_check_if_needed(tkr, df, True)
    year = pd.to_datetime(df['t'].iloc[0], unit='ms', utc=True).year
    path = f"stocks/{tkr}/{year}/{tkr}_{s:%Y-%m}.parquet"
    gcs.upload_parquet_df(df, path)
    report = {
        'ticker': tkr, 'start': str(s), 'end': str(e), 'rows': int(len(df)),
        'missing': missing[:50], 'price_flags': price_flags[:50], 'vol_flags': vol_flags[:50], 'yahoo_check': ychk,
    }
    gcs.upload_json(report, f"stocks/{tkr}/{year}/{tkr}_{s:%Y-%m}_validation.json")

async def backfill_stock_1m(gcs, settings, tickers: list[str]):
    async with PolygonClient(settings.polygon_key) as client:
        limiter = PaceLimiter(settings.stocks_rpm)
        sem = asyncio.Semaphore(settings.max_workers)
        tasks = []
        for tkr in tickers:
            for s,e in month_chunks(settings.start_date, settings.end_date):
                tasks.append(_guarded_one_month(gcs, settings, client, limiter, tkr, s, e, sem))
        await asyncio.gather(*tasks)

async def _guarded_one_month(gcs, settings, client, limiter, tkr, s, e, sem):
    async with sem:
        try:
            await _one_month_stock(gcs, settings, client, limiter, tkr, s, e)
        except Exception as ex:
            gcs.upload_json({'ticker': tkr, 'start': str(s), 'end': str(e), 'error': str(ex)}, f"stocks/{tkr}/_errors/{tkr}_{s:%Y-%m}.json")

async def top25_sp500(client: PolygonClient, sp500: set[str]) -> list[str]:
    snap = await client.snapshot_all()
    tickers = []
    for item in snap.get('tickers', []):
        t = item.get('ticker')
        if t in sp500:
            tickers.append((t, item.get('day', {}).get('v', 0)))
    tickers.sort(key=lambda x: x[1], reverse=True)
    return [t for t,_ in tickers[:25]]

async def backfill_nbbo_top25(gcs, settings, sp500: list[str]):
    async with PolygonClient(settings.polygon_key) as client:
        top = await top25_sp500(client, set(sp500))
        limiter = PaceLimiter(settings.stocks_rpm)
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=settings.nbbo_days_back)
        start_ns = int(start.value)
        end_ns = int(end.value)
        for t in top:
            await limiter.wait()
            quotes = await client.quotes(t, start_ns, end_ns, limit=50000)
            df = pd.DataFrame(quotes.get('results', []))
            if df.empty:
                continue
            year = pd.Timestamp.utcnow().year
            gcs.upload_parquet_df(df, f"stocks_nbbo/{t}/{year}/{t}_{start:%Y%m%d}_{end:%Y%m%d}.parquet")
EOF
}

fxdl() { cat > "$ROOT/src/downloaders/fx.py" <<'EOF'
import pandas as pd
from ..polygon_client import PolygonClient
from ..rate_limiter import PaceLimiter
from ..utils import month_chunks

async def backfill_fx_1m(gcs, settings, pairs: list[str]):
    async with PolygonClient(settings.polygon_key) as client:
        limiter = PaceLimiter(settings.basic_rpm)
        for p in pairs:
            for s,e in month_chunks(settings.start_date, settings.end_date):
                await limiter.wait()
                data = await client.aggs(p, 1, 'minute', str(s), str(e), adjusted=True)
                df = pd.DataFrame(data.get('results', []))
                if df.empty:
                    continue
                year = pd.to_datetime(df['t'].iloc[0], unit='ms', utc=True).year
                safe = p.replace(':','_')
                gcs.upload_parquet_df(df, f"fx/{safe}/{year}/{safe}_{s:%Y-%m}.parquet")
EOF
}

indicesdl() { cat > "$ROOT/src/downloaders/indices.py" <<'EOF'
import pandas as pd
from ..polygon_client import PolygonClient
from ..rate_limiter import PaceLimiter
from ..utils import month_chunks

async def backfill_indices_daily(gcs, settings, symbols: list[str]):
    async with PolygonClient(settings.polygon_key) as client:
        limiter = PaceLimiter(settings.basic_rpm)
        for sym in symbols:
            for s,e in month_chunks(settings.start_date, settings.end_date):
                await limiter.wait()
                data = await client.aggs(sym, 1, 'day', str(s), str(e), adjusted=True)
                df = pd.DataFrame(data.get('results', []))
                if df.empty:
                    continue
                year = pd.to_datetime(df['t'].iloc[0], unit='ms', utc=True).year
                safe = sym.replace(':','_')
                gcs.upload_parquet_df(df, f"indices/{safe}/{year}/{safe}_{s:%Y-%m}.parquet")
EOF
}

optionsdl() { cat > "$ROOT/src/downloaders/options.py" <<'EOF'
# Placeholder minimal implementation to keep within basic-tier limits.
# Fill with your own contract selection logic when ready.
async def backfill_options_sample(gcs, settings):
    if not settings.options['enabled']:
        return
    # Intentionally noop by default to avoid blasting API limits.
    return
EOF
}

futuresdl() { cat > "$ROOT/src/downloaders/futures.py" <<'EOF'
async def backfill_futures_guarded(gcs, settings):
    if not settings.futures['enabled']:
        return
    # Intentionally noop by default; enable when your plan supports it.
    return
EOF
}

orch() { cat > "$ROOT/src/orchestrator.py" <<'EOF'
import asyncio
from .config import load_settings, ASSETS
from .gcs import GCS
from .downloaders.stocks import backfill_stock_1m, backfill_nbbo_top25
from .downloaders.fx import backfill_fx_1m
from .downloaders.indices import backfill_indices_daily
from .downloaders.futures import backfill_futures_guarded
from .utils import fetch_sp500_from_wikipedia
from .polygon_client import PolygonClient

async def backfill_all():
    settings = load_settings()
    gcs = GCS(settings.bucket)

    try:
        async with PolygonClient(settings.polygon_key) as c:
            ms = await c.market_status()
            gcs.upload_json(ms, 'metadata/market_status.json')
    except Exception:
        pass

    sp500 = fetch_sp500_from_wikipedia()
    await backfill_stock_1m(gcs, settings, sp500 + ASSETS.get('etf_proxies', []))
    await backfill_nbbo_top25(gcs, settings, sp500)
    await backfill_fx_1m(gcs, settings, ASSETS.get('fx_pairs', []))
    await backfill_indices_daily(gcs, settings, ASSETS.get('indices_daily', []))
    await backfill_futures_guarded(gcs, settings)
EOF
}

run_init() { cat > "$ROOT/scripts/run_initial.py" <<'EOF'
import asyncio
from src.orchestrator import backfill_all

if __name__ == "__main__":
    asyncio.run(backfill_all())
EOF
}

run_incr() { cat > "$ROOT/scripts/run_incremental.py" <<'EOF'
import asyncio
from datetime import datetime, timedelta
from src.config import load_settings
from src.gcs import GCS
from src.downloaders.stocks import backfill_stock_1m, backfill_nbbo_top25
from src.downloaders.fx import backfill_fx_1m
from src.downloaders.indices import backfill_indices_daily
from src.utils import fetch_sp500_from_wikipedia

async def run():
    settings = load_settings()
    end = datetime.utcnow().date()
    start = end - timedelta(days=8)
    settings.start_date = str(start)
    settings.end_date = str(end)

    gcs = GCS(settings.bucket)
    sp500 = fetch_sp500_from_wikipedia()

    await backfill_stock_1m(gcs, settings, sp500)
    await backfill_nbbo_top25(gcs, settings, sp500)
    await backfill_fx_1m(gcs, settings, [])
    await backfill_indices_daily(gcs, settings, [])

if __name__ == "__main__":
    asyncio.run(run())
EOF
}

main() {
  mkdir -p "$ROOT"/{config,data,logs,scripts,src/downloaders}
  req
  settings
  assets
  initpy
  configpy
  gcspy
  ratelimiter
  polygon_client
  utilspy
  validatorspy
  stocksdl
  fxdl
  indicesdl
  optionsdl
  futuresdl
  orch
  run_init
  run_incr
  echo "Populated all files."
}

main
