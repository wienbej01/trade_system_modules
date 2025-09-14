#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/polygon-gcs-debug"

# Requirements with logging support
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
pyyaml==6.0.1
EOF
}

# Settings (unchanged)
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

# Assets (unchanged)
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

# Init py (unchanged)
initpy() { : > "$ROOT/src/__init__.py"; }

# Config with logging (add logging import)
configpy() { cat > "$ROOT/src/config.py" <<'EOF'
import os, yaml, logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    options: Optional[Dict[str, Any]] = None
    futures: Optional[Dict[str, Any]] = None
    validate_missing: bool = True
    validate_outliers: bool = True
    outlier: Dict[str, float] = None

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_settings():
    s = load_yaml('config/settings.yaml')
    logger.info(f"Loading settings: start={s['start_date']}, end={s['end_date']}, rpm_stocks={os.environ.get('STOCKS_RPM', 120)}, rpm_basic={os.environ.get('BASIC_ASSETS_RPM', 5)}")
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
        options=s.get('options', {}),
        futures=s.get('futures', {}),
        validate_missing=bool(s.get('validate_missing', True)),
        validate_outliers=bool(s.get('validate_outliers', True)),
        outlier=s.get('outlier', {'price_pct_jump': 0.2, 'vol_sigma': 6}),
    )

ASSETS = load_yaml('config/assets.yaml')
logger.info(f"Loaded assets: {len(ASSETS.get('fx_pairs', []))} FX pairs, {len(ASSETS.get('etf_proxies', []))} ETF proxies")
EOF
}

# GCS with existence logging
gcspy() { cat > "$ROOT/src/gcs.py" <<'EOF'
import logging
from google.cloud import storage
from io import BytesIO
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class GCS:
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        logger.info(f"Initialized GCS client for bucket: {bucket_name}")

    def exists(self, path: str) -> bool:
        exists = self.bucket.blob(path).exists()
        logger.info(f"GCS exists check for {path}: {exists}")
        return exists

    def upload_parquet_df(self, df: pd.DataFrame, path: str):
        if self.exists(path):
            logger.info(f"Skipping upload to {path} - file already exists")
            return
        logger.info(f"Uploading {len(df)} rows to {path}")
        table = pa.Table.from_pandas(df, preserve_index=False)
        buf = BytesIO()
        pq.write_table(table, buf, compression='snappy')
        blob = self.bucket.blob(path)
        blob.upload_from_string(buf.getvalue(), content_type='application/octet-stream')
        logger.info(f"Successfully uploaded to {path}")

    def upload_json(self, obj: dict, path: str):
        if self.exists(path):
            logger.info(f"Skipping JSON upload to {path} - file already exists")
            return
        import json
        blob = self.bucket.blob(path)
        blob.upload_from_string(json.dumps(obj, indent=2), content_type='application/json')
        logger.info(f"Uploaded JSON to {path}")
EOF
}

# Rate limiter (unchanged)
ratelimiter() { cat > "$ROOT/src/rate_limiter.py" <<'EOF'
import asyncio, time, logging

logger = logging.getLogger(__name__)

class PaceLimiter:
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(1, rpm)
        self._last = 0.0
        self._lock = asyncio.Lock()
        logger.info(f"Initialized PaceLimiter with RPM: {rpm}, interval: {self.min_interval:.2f}s")

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                logger.debug(f"Rate limit wait: {wait:.2f}s")
                await asyncio.sleep(wait)
            self._last = time.monotonic()
EOF
}

# Polygon client with response logging
polygon_client() { cat > "$ROOT/src/polygon_client.py" <<'EOF'
import aiohttp, asyncio, backoff, logging
import pandas as pd
from urllib.parse import urlencode
import json

logger = logging.getLogger(__name__)
API = 'https://api.polygon.io'

class PolygonClient:
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
        logger.info(f"Initialized PolygonClient with API key: {api_key[:8]}...")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        logger.info("Opened PolygonClient session")
        return self

    async def __aexit__(self, *exc):
        await self.session.close()
        logger.info("Closed PolygonClient session")

    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=5)
    async def _get(self, path: str, params: dict):
        params = {**params, 'apiKey': self.api_key}
        if path.startswith('http'):
            url = path
        else:
            url = f"{API}{path}?{urlencode(params, doseq=True)}"
        logger.debug(f"API GET: {url[:100]}...")
        async with self.session.get(url) as resp:
            logger.debug(f"API response status: {resp.status}")
            if resp.status == 429:
                logger.warning("Rate limited (429)")
                raise aiohttp.ClientError('rate limited')
            if resp.status == 403:
                logger.error(f"Forbidden (403) for {url[:100]}...")
            resp.raise_for_status()
            data = await resp.json()
            logger.debug(f"API response keys: {list(data.keys())}")
            return data

    async def aggs(self, ticker: str, multiplier: int, timespan: str, start: str, end: str, limit: int = 50000, adjusted: bool = True, sort: str = 'asc'):
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
        logger.debug(f"Aggs request for {ticker}: {start} to {end}")
        return await self._get(path, {'limit': limit, 'adjusted': str(adjusted).lower(), 'sort': sort})

    async def quotes(self, ticker: str, start_ts_ns: int, end_ts_ns: int, limit: int = 50000):
        path = f"/v3/quotes/{ticker}"
        logger.info(f"Quotes request for {ticker}: start_ns={start_ts_ns}, end_ns={end_ts_ns}")
        logger.info(f"Timestamps decode: start={pd.Timestamp(start_ts_ns, unit='ns')}, end={pd.Timestamp(end_ts_ns, unit='ns')}")
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

# Utils with logging
utilspy() { cat > "$ROOT/src/utils.py" <<'EOF'
import logging
from datetime import datetime
import pandas as pd
import pytz
import pandas_market_calendars as mcal
import yfinance as yf

logger = logging.getLogger(__name__)
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

async def fetch_active_stocks(client: PolygonClient, market: str = 'stocks', limit_per_page: int = 1000) -> list[str]:
    """Fetch all active stock tickers from Polygon.io API with pagination."""
    all_tickers = []
    next_url = None
    page = 0
    logger.info(f"Fetching active {market} tickers from Polygon.io...")
    
    while True:
        page += 1
        params = {'market': market, 'active': 'true', 'limit': limit_per_page}
        if next_url:
            data = await client._get(next_url, {})
        else:
            data = await client.list_tickers(**params)
        
        results = data.get('results', [])
        for ticker in results:
            symbol = ticker.get('ticker')
            if symbol:
                all_tickers.append(symbol)
        
        logger.info(f"Page {page}: fetched {len(results)} tickers (total: {len(all_tickers)})")
        
        next_url = data.get('next_url')
        if not next_url:
            break
    
    logger.info(f"Fetched {len(all_tickers)} active {market} tickers from Polygon.io")
    return all_tickers

def fetch_sp500_fallback() -> list[str]:
    """Fallback hardcoded S&P500 list for testing."""
    fallback = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'JPM', 'V', 'PG', 'HD', 'MA', 'CVX', 'ABBV', 'PFE', 'BAC', 'KO',
        'CRM', 'ADBE', 'NFLX', 'COST', 'TMO', 'AVGO', 'LIN', 'WMT', 'MRK', 'ACN',
        'TMUS', 'DHR', 'ABT', 'VZ', 'NEE', 'TXN', 'UNH', 'CSCO', 'WFC', 'XOM', 'ORCL'
    ]
    logger.info(f"Using fallback S&P500 list: {len(fallback)} symbols")
    return fallback

def fetch_sp500_from_wikipedia() -> list[str]:
    """Legacy Wikipedia fetch - use only if Polygon API unavailable."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        import requests
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        df = tables[0]
        syms = sorted(set(df['Symbol'].astype(str).str.replace('.', '-', regex=False)))
        logger.info(f"Fetched {len(syms)} S&P500 symbols from Wikipedia")
        return syms
    except Exception as ex:
        logger.error(f"Wikipedia fetch failed: {ex}. Returning fallback.")
        return fetch_sp500_fallback()

def yahoo_daily_ohlc(ticker: str, dt: pd.Timestamp):
    logger.info(f"yfinance call for {ticker} around {dt.date()}")
    try:
        start = (dt.tz_convert(NY) - pd.Timedelta(days=1)).date()
        end = (dt.tz_convert(NY) + pd.Timedelta(days=2)).date()
        y = yf.Ticker(ticker)
        hist = y.history(interval='1d', start=start, end=end, auto_adjust=False)
        key = str(dt.date())
        if key in hist.index.astype(str):
            row = hist.loc[key]
            logger.info(f"yfinance success for {ticker}: low={row['Low']:.2f}, high={row['High']:.2f}")
            return row
        else:
            logger.warning(f"yfinance no data for {ticker} on {key}")
            return None
    except Exception as ex:
        logger.error(f"yfinance error for {ticker}: {ex}")
        return None
EOF
}

# Validators with yfinance logging
validatorspy() { cat > "$ROOT/src/validators.py" <<'EOF'
import logging, pandas as pd
from .utils import to_et, session_label, expected_minutes_for_day, yahoo_daily_ohlc

logger = logging.getLogger(__name__)

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
    if missing:
        logger.warning(f"Found {len(missing)} missing minutes")
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
    if price_flags or vol_flags:
        logger.warning(f"Found {len(price_flags)} price outliers, {len(vol_flags)} volume outliers")
    return price_flags, vol_flags

def yahoo_sanity_check_if_needed(ticker: str, df: pd.DataFrame, anomalies: bool):
    if not anomalies or df.empty:
        logger.debug(f"No yfinance check needed for {ticker} (no anomalies)")
        return None
    day = pd.to_datetime(df['t'].iloc[0], unit='ms', utc=True)
    logger.info(f"Performing yfinance sanity check for {ticker} due to anomalies")
    row = yahoo_daily_ohlc(ticker, day)
    if row is None or len(row) == 0:
        return {'status': 'yahoo_missing'}
    lo, hi = float(row['Low']), float(row['High'])
    ok = (df['l'].min() >= lo - 1e-6) and (df['h'].max() <= hi + 1e-6)
    status = 'ok' if ok else 'mismatch'
    logger.info(f"yfinance sanity check for {ticker}: status={status}, yahoo_range=[{lo:.2f}, {hi:.2f}]")
    return {'status': status, 'yahoo_low': lo, 'yahoo_high': hi}
EOF
}

# Stocks downloader with validation logging
stocksdl() { cat > "$ROOT/src/downloaders/stocks.py" <<'EOF'
import asyncio, logging, pandas as pd
from ..polygon_client import PolygonClient
from ..rate_limiter import PaceLimiter
from ..validators import label_sessions, find_missing_minutes, find_outliers, yahoo_sanity_check_if_needed
from ..utils import month_chunks
from tqdm import tqdm

logger = logging.getLogger(__name__)
FIELDS = ['t','o','h','l','c','v','vw','n']

async def fetch_stock_month(client: PolygonClient, limiter: PaceLimiter, ticker: str, start: str, end: str):
    await limiter.wait()
    logger.debug(f"Fetching aggs for {ticker}: {start} to {end}")
    data = await client.aggs(ticker=ticker, multiplier=1, timespan='minute', start=start, end=end, adjusted=True, sort='asc')
    df = pd.DataFrame(data.get('results', []))
    if not df.empty:
        df = df[[c for c in FIELDS if c in df.columns]]
        logger.debug(f"Fetched {len(df)} rows for {ticker}")
    else:
        logger.warning(f"No data for {ticker}: {start} to {end}")
    return df

async def _one_month_stock(gcs, settings, client, limiter, tkr, s, e):
    start_time = asyncio.get_event_loop().time()
    df = await fetch_stock_month(client, limiter, tkr, str(s), str(e))
    if df.empty:
        logger.warning(f"Empty data for {tkr} {s} to {e}, skipping")
        return
    df = label_sessions(df)
    missing = find_missing_minutes(df, include_extended=settings.include_extended_hours) if settings.validate_missing else []
    price_flags, vol_flags = find_outliers(df, settings.outlier['price_pct_jump'], settings.outlier['vol_sigma']) if settings.validate_outliers else ([],[])
    anomalies = bool(missing or price_flags or vol_flags)
    ychk = None
    if settings.yahoo_validate_on_anomaly and anomalies:
        logger.info(f"Anomalies detected for {tkr} {s}, running yfinance check")
        ychk = yahoo_sanity_check_if_needed(tkr, df, True)
    else:
        logger.debug(f"No anomalies for {tkr} {s}, skipping yfinance")
    year = pd.to_datetime(df['t'].iloc[0], unit='ms', utc=True).year
    path = f"stocks/{tkr}/{year}/{tkr}_{s:%Y-%m}.parquet"
    gcs.upload_parquet_df(df, path)
    report = {
        'ticker': tkr, 'start': str(s), 'end': str(e), 'rows': int(len(df)),
        'missing': missing[:50], 'price_flags': price_flags[:50], 'vol_flags': vol_flags[:50], 'yahoo_check': ychk,
    }
    gcs.upload_json(report, f"stocks/{tkr}/{year}/{tkr}_{s:%Y-%m}_validation.json")
    duration = asyncio.get_event_loop().time() - start_time
    logger.info(f"[stocks] OK   {tkr} {s:%Y-%m} rows={len(df)} duration={duration:.1f}s")

async def backfill_stock_1m(gcs, settings, tickers: list[str]):
    start_time = asyncio.get_event_loop().time()
    logger.info(f"Starting stock backfill for {len(tickers)} tickers from {settings.start_date} to {settings.end_date}")
    async with PolygonClient(settings.polygon_key) as client:
        limiter = PaceLimiter(settings.stocks_rpm)
        sem = asyncio.Semaphore(settings.max_workers)
        tasks = []
        total_chunks = sum(len(list(month_chunks(settings.start_date, settings.end_date))) for _ in tickers)
        logger.info(f"Total ticker-months to process: ~{total_chunks}")
        with tqdm(total=total_chunks, desc="stocks 1m") as pbar:
            for tkr in tickers:
                for s,e in month_chunks(settings.start_date, settings.end_date):
                    task = _guarded_one_month(gcs, settings, client, limiter, tkr, s, e, sem, pbar)
                    tasks.append(task)
            await asyncio.gather(*tasks)
    duration = asyncio.get_event_loop().time() - start_time
    logger.info(f"Stock backfill completed in {duration/3600:.1f} hours")

async def _guarded_one_month(gcs, settings, client, limiter, tkr, s, e, sem, pbar):
    async with sem:
        try:
            await _one_month_stock(gcs, settings, client, limiter, tkr, s, e)
        except Exception as ex:
            logger.error(f"Error for {tkr} {s} to {e}: {ex}")
            gcs.upload_json({'ticker': tkr, 'start': str(s), 'end': str(e), 'error': str(ex)}, f"stocks/{tkr}/_errors/{tkr}_{s:%Y-%m}.json")
        finally:
            pbar.update(1)

async def top25_sp500(client: PolygonClient, sp500: set[str]) -> list[str]:
    logger.info("Fetching top-25 S&P500 by day volume...")
    snap = await client.snapshot_all()
    tickers = []
    for item in snap.get('tickers', []):
        t = item.get('ticker')
        if t in sp500:
            tickers.append((t, item.get('day', {}).get('v', 0)))
    tickers.sort(key=lambda x: x[1], reverse=True)
    top = [t for t,_ in tickers[:25]]
    logger.info(f"[nbbo] symbols: 25 → {', '.join(top[:5])}...")
    return top

async def backfill_nbbo_top25(gcs, settings, sp500: list[str]):
    start_time = asyncio.get_event_loop().time()
    logger.info(f"[orchestrator] NBBO (Top-25 S&P500)…")
    async with PolygonClient(settings.polygon_key) as client:
        top = await top25_sp500(client, set(sp500))
        limiter = PaceLimiter(settings.stocks_rpm)
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=settings.nbbo_days_back)
        start_ns = int(start.value)  # Proper nanoseconds since epoch
        end_ns = int(end.value)
        logger.info(f"NBBO timestamp range: start={start}, end={end}, start_ns={start_ns}, end_ns={end_ns}")
        logger.info(f"Timestamps decode: start={pd.Timestamp(start_ns, unit='ns')}, end={pd.Timestamp(end_ns, unit='ns')}")
        for t in top:
            await limiter.wait()
            try:
                quotes = await client.quotes(t, start_ns, end_ns, limit=50000)
                df = pd.DataFrame(quotes.get('results', []))
                if df.empty:
                    logger.warning(f"No NBBO quotes for {t}")
                    continue
                year = pd.Timestamp.utcnow().year
                path = f"stocks_nbbo/{t}/{year}/{t}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
                gcs.upload_parquet_df(df, path)
                logger.info(f"NBBO OK {t}: {len(df)} quotes")
            except Exception as ex:
                logger.error(f"NBBO error for {t}: {ex}")
                continue
    duration = asyncio.get_event_loop().time() - start_time
    logger.info(f"NBBO backfill completed in {duration/60:.1f} minutes")
EOF
}

# FX (add logging)
fxdl() { cat > "$ROOT/src/downloaders/fx.py" <<'EOF'
import asyncio, logging, pandas as pd
from ..polygon_client import PolygonClient
from ..rate_limiter import PaceLimiter
from ..utils import month_chunks

logger = logging.getLogger(__name__)

async def backfill_fx_1m(gcs, settings, pairs: list[str]):
    start_time = asyncio.get_event_loop().time()
    logger.info(f"Starting FX backfill for {len(pairs)} pairs")
    async with PolygonClient(settings.polygon_key) as client:
        limiter = PaceLimiter(settings.basic_rpm)
        for p in pairs:
            logger.info(f"Processing FX pair: {p}")
            for s,e in month_chunks(settings.start_date, settings.end_date):
                await limiter.wait()
                data = await client.aggs(p, 1, 'minute', str(s), str(e), adjusted=True)
                df = pd.DataFrame(data.get('results', []))
                if df.empty:
                    continue
                year = pd.to_datetime(df['t'].iloc[0], unit='ms', utc=True).year
                safe = p.replace(':','_')
                path = f"fx/{safe}/{year}/{safe}_{s:%Y-%m}.parquet"
                gcs.upload_parquet_df(df, path)
                logger.info(f"FX OK {p} {s:%Y-%m}: {len(df)} rows")
    duration = asyncio.get_event_loop().time() - start_time
    logger.info(f"FX backfill completed in {duration/60:.1f} minutes")
EOF
}

# Indices (add logging)
indicesdl() { cat > "$ROOT/src/downloaders/indices.py" <<'EOF'
import asyncio, logging, pandas as pd
from ..polygon_client import PolygonClient
from ..rate_limiter import PaceLimiter
from ..utils import month_chunks

logger = logging.getLogger(__name__)

async def backfill_indices_daily(gcs, settings, symbols: list[str]):
    start_time = asyncio.get_event_loop().time()
    logger.info(f"Starting indices backfill for {len(symbols)} symbols")
    async with PolygonClient(settings.polygon_key) as client:
        limiter = PaceLimiter(settings.basic_rpm)
        for sym in symbols:
            logger.info(f"Processing index: {sym}")
            for s,e in month_chunks(settings.start_date, settings.end_date):
                await limiter.wait()
                data = await client.aggs(sym, 1, 'day', str(s), str(e), adjusted=True)
                df = pd.DataFrame(data.get('results', []))
                if df.empty:
                    continue
                year = pd.to_datetime(df['t'].iloc[0], unit='ms', utc=True).year
                safe = sym.replace(':','_')
                path = f"indices/{safe}/{year}/{safe}_{s:%Y-%m}.parquet"
                gcs.upload_parquet_df(df, path)
                logger.info(f"Index OK {sym} {s:%Y-%m}: {len(df)} rows")
    duration = asyncio.get_event_loop().time() - start_time
    logger.info(f"Indices backfill completed in {duration/60:.1f} minutes")
EOF
}

# Options and Futures (unchanged, with log)
optionsdl() { cat > "$ROOT/src/downloaders/options.py" <<'EOF'
import logging

logger = logging.getLogger(__name__)

async def backfill_options_sample(gcs, settings):
    if not settings.options or not settings.options.get('enabled', False):
        logger.info("Options disabled, skipping")
        return
    logger.warning("Options backfill not implemented (API limits)")
    return
EOF
}

futuresdl() { cat > "$ROOT/src/downloaders/futures.py" <<'EOF'
import logging

logger = logging.getLogger(__name__)

async def backfill_futures_guarded(gcs, settings):
    if not settings.futures or not settings.futures.get('enabled', False):
        logger.info("Futures disabled, skipping")
        return
    logger.warning("Futures backfill not implemented (API limits)")
    return
EOF
}

# Orchestrator with try-catch and timing
orch() { cat > "$ROOT/src/orchestrator.py" <<'EOF'
import asyncio, logging
from .config import load_settings, ASSETS
from .gcs import GCS
from .downloaders.stocks import backfill_stock_1m, backfill_nbbo_top25
from .downloaders.fx import backfill_fx_1m
from .downloaders.indices import backfill_indices_daily
from .downloaders.futures import backfill_futures_guarded
from .downloaders.options import backfill_options_sample
from .utils import fetch_sp500_from_wikipedia
from .polygon_client import PolygonClient

logger = logging.getLogger(__name__)

async def backfill_all():
    overall_start = asyncio.get_event_loop().time()
    settings = load_settings()
    gcs = GCS(settings.bucket)
    logger.info("=== Starting full backfill orchestration ===")

    # Market status
    try:
        async with PolygonClient(settings.polygon_key) as c:
            ms = await c.market_status()
            gcs.upload_json(ms, 'metadata/market_status.json')
            logger.info("Market status uploaded")
    except Exception as ex:
        logger.error(f"Market status failed: {ex}")

    # Fetch S&P500
    try:
        sp500_start = asyncio.get_event_loop().time()
        sp500 = fetch_sp500_from_wikipedia()
        sp500_duration = asyncio.get_event_loop().time() - sp500_start
        logger.info(f"S&P500 fetch completed ({sp500_duration:.1f}s): {len(sp500)} symbols")
    except Exception as ex:
        logger.error(f"S&P500 fetch FAILED: {ex}")
        sp500 = []  # Continue with empty list

    # Stocks
    try:
        stocks_start = asyncio.get_event_loop().time()
        await backfill_stock_1m(gcs, settings, sp500 + ASSETS.get('etf_proxies', []))
        stocks_duration = asyncio.get_event_loop().time() - stocks_start
        logger.info(f"Stocks backfill SUCCESS in {stocks_duration/3600:.1f} hours")
    except Exception as ex:
        logger.error(f"Stocks backfill FAILED: {ex}")

    # NBBO
    try:
        nbbo_start = asyncio.get_event_loop().time()
        await backfill_nbbo_top25(gcs, settings, sp500)
        nbbo_duration = asyncio.get_event_loop().time() - nbbo_start
        logger.info(f"NBBO backfill SUCCESS in {nbbo_duration/60:.1f} minutes")
    except Exception as ex:
        logger.error(f"NBBO backfill FAILED: {ex}")

    # FX
    try:
        fx_start = asyncio.get_event_loop().time()
        await backfill_fx_1m(gcs, settings, ASSETS.get('fx_pairs', []))
        fx_duration = asyncio.get_event_loop().time() - fx_start
        logger.info(f"FX backfill SUCCESS in {fx_duration/60:.1f} minutes")
    except Exception as ex:
        logger.error(f"FX backfill FAILED: {ex}")

    # Indices
    try:
        indices_start = asyncio.get_event_loop().time()
        await backfill_indices_daily(gcs, settings, ASSETS.get('indices_daily', []))
        indices_duration = asyncio.get_event_loop().time() - indices_start
        logger.info(f"Indices backfill SUCCESS in {indices_duration/60:.1f} minutes")
    except Exception as ex:
        logger.error(f"Indices backfill FAILED: {ex}")

    # Options
    try:
        await backfill_options_sample(gcs, settings)
        logger.info("Options backfill completed")
    except Exception as ex:
        logger.error(f"Options backfill FAILED: {ex}")

    # Futures
    try:
        await backfill_futures_guarded(gcs, settings)
        logger.info("Futures backfill completed")
    except Exception as ex:
        logger.error(f"Futures backfill FAILED: {ex}")

    overall_duration = asyncio.get_event_loop().time() - overall_start
    logger.info(f"=== Full backfill completed in {overall_duration/3600:.1f} hours ===")
EOF
}

# Run initial with logging
run_init() { cat > "$ROOT/scripts/run_initial.py" <<'EOF'
import asyncio, logging, sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.orchestrator import backfill_all

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting initial backfill run")
        asyncio.run(backfill_all())
    except KeyboardInterrupt:
        logger.info("Run interrupted by user")
    except Exception as e:
        logger.error(f"Run failed: {e}", exc_info=True)
        raise
EOF
}

# Run incremental (similar logging)
run_incr() { cat > "$ROOT/scripts/run_incremental.py" <<'EOF'
import asyncio, logging, sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_settings
from src.gcs import GCS
from src.downloaders.stocks import backfill_stock_1m, backfill_nbbo_top25
from src.downloaders.fx import backfill_fx_1m
from src.downloaders.indices import backfill_indices_daily
from src.utils import fetch_sp500_from_wikipedia

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

async def run():
    settings = load_settings()
    end = datetime.utcnow().date()
    start = end - timedelta(days=8)
    settings.start_date = str(start)
    settings.end_date = str(end)
    logger.info(f"Incremental run: {start} to {end}")

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
  echo "Debug version populated in $ROOT"
  echo "To run: cd $ROOT && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && export POLYGON_API_KEY=your_key && export GCS_BUCKET=your_bucket && nohup python scripts/run_initial.py > debug.log 2>&1 &"
}

main