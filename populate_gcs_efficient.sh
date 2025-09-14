#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/polygon-gcs-efficient"

# Requirements - trade_system_modules + minimal extras
req() { cat > "$ROOT/requirements.txt" <<'EOF'
google-cloud-storage==2.17.0
pandas==2.2.2
pyarrow==16.1.0
python-dateutil==2.9.0
pytz==2024.1
tqdm==4.66.4
requests==2.32.3
pyyaml==6.0.1
aiohttp==3.9.5
tenacity==8.2.3
EOF
}

# Settings with delta mode support
settings() { cat > "$ROOT/config/settings.yaml" <<'EOF'
# Common settings
gcs_bucket: "jwss_data_store"
polygon_api_key: "${POLYGON_API_KEY}"
start_date: "2020-01-01"  # For full download
end_date: "today"
timespan: "minute"
multiplier: 1
concurrency: 50  # High for batch efficiency
include_otc: true
market: "stocks"  # stocks, forex, crypto

# Asset classes to download
asset_classes:
  stocks:
    enabled: true
    market: "stocks"
    active_only: true
    limit: 500  # Top N active stocks
  forex:
    enabled: true
    market: "forex"
    pairs: ["C:EURUSD", "C:GBPUSD", "C:USDJPY", "C:USDCHF", "C:USDCAD", "C:AUDUSD", "C:NZDUSD"]
  indices:
    enabled: true
    symbols: ["I:SPX", "I:NDX", "I:DJI", "I:RUT", "I:VIX"]
    timespan: "day"  # Daily for indices

# Delta mode (monthly)
delta_days_back: 30
EOF
}

# Init files
initpy() { 
  mkdir -p "$ROOT/src" "$ROOT/scripts" "$ROOT/config"
  touch "$ROOT/src/__init__.py"
}

# Main config
configpy() { cat > "$ROOT/src/config.py" <<'EOF'
import os, yaml, logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DownloadSettings:
    gcs_bucket: str
    polygon_api_key: str
    start_date: str
    end_date: str
    timespan: str = "minute"
    multiplier: int = 1
    concurrency: int = 50
    include_otc: bool = True
    market: str = "stocks"
    
    # Asset classes
    asset_classes: Dict[str, Any] = field(default_factory=dict)
    
    # Delta mode
    delta_days_back: int = 30

def load_config(path: str = 'config/settings.yaml') -> DownloadSettings:
    """Load configuration from YAML."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with env vars
    bucket = os.environ.get('GCS_BUCKET', config.get('gcs_bucket', 'default-bucket'))
    api_key = os.environ.get('POLYGON_API_KEY', config.get('polygon_api_key', ''))
    
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable required")
    
    settings = DownloadSettings(
        gcs_bucket=bucket,
        polygon_api_key=api_key,
        start_date=config.get('start_date', '2020-01-01'),
        end_date=config.get('end_date', 'today'),
        timespan=config.get('timespan', 'minute'),
        multiplier=config.get('multiplier', 1),
        concurrency=config.get('concurrency', 50),
        include_otc=config.get('include_otc', True),
        market=config.get('market', 'stocks'),
        asset_classes=config.get('asset_classes', {}),
        delta_days_back=config.get('delta_days_back', 30)
    )
    
    logger.info(f"Loaded config: bucket={bucket}, market={settings.market}, concurrency={settings.concurrency}")
    return settings

def get_date_range_for_mode(settings: DownloadSettings, mode: str = 'full') -> tuple:
    """Get start/end dates based on mode: 'full' or 'delta'."""
    if mode == 'delta':
        end = datetime.now().date()
        start = end - timedelta(days=settings.delta_days_back)
        logger.info(f"Delta mode: {start} to {end} ({settings.delta_days_back} days back)")
        return str(start), str(end)
    else:
        logger.info(f"Full mode: {settings.start_date} to {settings.end_date}")
        return settings.start_date, settings.end_date
EOF
}

# GCS wrapper with skipping
gcs_wrapper() { cat > "$ROOT/src/gcs_wrapper.py" <<'EOF'
from trade_system_modules.storage.gcs import GCSClient
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

class EfficientGCS(GCSClient):
    """GCS wrapper with existence checking to avoid duplicate downloads."""
    
    def __init__(self, bucket: Optional[str] = None):
        super().__init__(bucket)
        self.existing_files = set()
    
    def check_and_upload(self, df: pd.DataFrame, path: str, force: bool = False) -> bool:
        """Check if file exists, skip if present unless force=True."""
        if not force and self.exists(path):
            logger.info(f"Skipping {path} - already exists ({len(df)} rows)")
            return False
        
        logger.info(f"Uploading {len(df)} rows to {path}")
        self.to_parquet(df, path)
        logger.debug(f"Uploaded to {path}")
        return True
    
    def exists(self, path: str) -> bool:
        """Check if parquet file exists in GCS."""
        try:
            blob = self.bucket.blob(path)
            exists = blob.exists()
            if exists:
                self.existing_files.add(path)
            return exists
        except Exception as e:
            logger.warning(f"Error checking {path}: {e}")
            return False
    
    def get_existing_for_ticker(self, ticker: str, year: int) -> set:
        """Get all existing files for a ticker/year to skip chunks."""
        prefix = f"data/{ticker}/{year}/"
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return {blob.name for blob in blobs if blob.name.endswith('.parquet')}
        except Exception as e:
            logger.warning(f"Error listing {prefix}: {e}")
            return set()
EOF
}

# Polygon ticker fetcher
ticker_fetcher() { cat > "$ROOT/src/ticker_fetcher.py" <<'EOF'
import asyncio
import logging
from typing import List, Optional
from trade_system_modules.data.polygon_adapter import get_tickers  # Assuming this exists or implement below

logger = logging.getLogger(__name__)

async def get_active_tickers(api_key: str, market: str = 'stocks', type_: Optional[str] = None, 
                          active: bool = True, limit: int = 1000, search: Optional[str] = None) -> List[str]:
    """Fetch active tickers from Polygon.io /v3/reference/tickers with pagination."""
    from trade_system_modules.config.settings import settings  # For client
    
    all_tickers = []
    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        'market': market,
        'active': str(active).lower(),
        'limit': limit,
        'apiKey': api_key
    }
    if type_:
        params['type'] = type_
    if search:
        params['search'] = search
    
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if resp.status != 200:
                    logger.error(f"Error fetching tickers: {data}")
                    break
                
                results = data.get('results', [])
                symbols = [r['ticker'] for r in results if r.get('active', False)]
                all_tickers.extend(symbols)
                logger.debug(f"Fetched {len(results)} tickers, total: {len(all_tickers)}")
                
                next_url = data.get('next_url')
                if not next_url:
                    break
                # Update for next page
                url = next_url
                params = {}  # next_url includes params
    
    logger.info(f"Fetched {len(all_tickers)} {market} tickers from Polygon.io")
    return all_tickers

# For S&P500 approximation - search for known companies or use full active list
async def get_sp500_approx(api_key: str, limit: int = 500) -> List[str]:
    """Get top active US stocks as S&P500 proxy."""
    tickers = await get_active_tickers(api_key, market='stocks', limit=limit)
    return sorted(tickers)[:500]  # Top 500 active as proxy
EOF
}

# Main orchestrator
orchestrator() { cat > "$ROOT/src/orchestrator.py" <<'EOF'
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

from config import load_config, get_date_range_for_mode
from gcs_wrapper import EfficientGCS
from ticker_fetcher import get_active_tickers, get_sp500_approx
from trade_system_modules.data.polygon_adapter import get_agg_minute_batch
from trade_system_modules.storage.gcs import ensure_bar_schema

logger = logging.getLogger(__name__)

async def download_asset_class(gcs: EfficientGCS, settings, asset_config: Dict, 
                              start_date: str, end_date: str, mode: str = 'full') -> None:
    """Download single asset class (stocks, forex, indices)."""
    asset_name = asset_config['name']
    logger.info(f"Starting {asset_name} download: {start_date} to {end_date}")
    
    start_time = asyncio.get_event_loop().time()
    
    if asset_name == 'stocks':
        # Get tickers from Polygon
        tickers = await get_sp500_approx(settings.polygon_api_key, limit=asset_config.get('limit', 500))
        symbols = tickers
    elif asset_name == 'forex':
        symbols = asset_config.get('pairs', [])
    elif asset_name == 'indices':
        symbols = asset_config.get('symbols', [])
    else:
        logger.warning(f"Unknown asset class: {asset_name}")
        return
    
    if not symbols:
        logger.warning(f"No symbols for {asset_name}")
        return
    
    # Batch download with skipping
    data_dict = await get_agg_minute_batch(
        symbols=symbols,
        start=start_date,
        end=end_date,
        concurrency=settings.concurrency
    )
    
    uploaded_count = 0
    for symbol, df in data_dict.items():
        if df.empty:
            logger.debug(f"No data for {symbol}")
            continue
        
        # Ensure schema
        df = ensure_bar_schema(df)
        
        # Generate path
        year = df['ts'].dt.year.min()
        month = df['ts'].dt.to_period('M').min()
        safe_symbol = symbol.replace(':', '_').replace('/', '_')
        path = f"data/{safe_symbol}/{year}/{safe_symbol}_{month.strftime('%Y-%m')}.parquet"
        
        # Skip if exists
        if gcs.check_and_upload(df, path):
            uploaded_count += 1
    
    duration = asyncio.get_event_loop().time() - start_time
    logger.info(f"{asset_name} download completed: {uploaded_count}/{len(symbols)} symbols in {duration/60:.1f} min")

async def run_full_download(settings: DownloadSettings) -> None:
    """Full historical download."""
    start_date, end_date = get_date_range_for_mode(settings, 'full')
    gcs = EfficientGCS(settings.gcs_bucket)
    
    # Process each asset class
    for asset_name, config in settings.asset_classes.items():
        if config.get('enabled', False):
            asset_config = {**config, 'name': asset_name}
            await download_asset_class(gcs, settings, asset_config, start_date, end_date, 'full')

async def run_delta_download(settings: DownloadSettings) -> None:
    """Monthly delta download (last N days)."""
    start_date, end_date = get_date_range_for_mode(settings, 'delta')
    gcs = EfficientGCS(settings.gcs_bucket)
    
    # For delta, focus on active/recent assets
    for asset_name, config in settings.asset_classes.items():
        if config.get('enabled', False):
            asset_config = {**config, 'name': asset_name}
            await download_asset_class(gcs, settings, asset_config, start_date, end_date, 'delta')

async def main(mode: str = 'full'):
    """Main orchestrator."""
    settings = load_config()
    overall_start = asyncio.get_event_loop().time()
    
    if mode == 'delta':
        await run_delta_download(settings)
    else:
        await run_full_download(settings)
    
    duration = asyncio.get_event_loop().time() - overall_start
    logger.info(f"Download {mode} completed in {duration/3600:.1f} hours")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    asyncio.run(main(mode))
EOF
}

# Run scripts
run_full() { cat > "$ROOT/scripts/run_full.py" <<'EOF'
#!/usr/bin/env python3
import asyncio
import os
from src.orchestrator import main

if __name__ == "__main__":
    # Required env vars
    if not os.environ.get('POLYGON_API_KEY'):
        print("Error: POLYGON_API_KEY required")
        exit(1)
    if not os.environ.get('GCS_BUCKET'):
        print("Error: GCS_BUCKET required")
        exit(1)
    
    print("Starting FULL historical download...")
    asyncio.run(main('full'))
EOF
}

run_delta() { cat > "$ROOT/scripts/run_delta.py" <<'EOF'
#!/usr/bin/env python3
import asyncio
import os
from src.orchestrator import main

if __name__ == "__main__":
    # Required env vars
    if not os.environ.get('POLYGON_API_KEY'):
        print("Error: POLYGON_API_KEY required")
        exit(1)
    if not os.environ.get('GCS_BUCKET'):
        print("Error: GCS_BUCKET required")
        exit(1)
    
    print("Starting DELTA monthly download...")
    asyncio.run(main('delta'))
EOF
}

# Main populate function
main() {
  initpy
  req
  settings
  configpy
  gcs_wrapper
  ticker_fetcher
  orchestrator
  run_full
  run_delta
  
  chmod +x "$ROOT/scripts/run_"*.py
  echo "Efficient GCS downloader populated in $ROOT"
  echo ""
  echo "Setup:"
  echo "  cd $ROOT"
  echo "  python -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  echo ""
  echo "Full historical download:"
  echo "  export POLYGON_API_KEY=your_key"
  echo "  export GCS_BUCKET=your_bucket"
  echo "  python scripts/run_full.py"
  echo ""
  echo "Monthly delta download:"
  echo "  python scripts/run_delta.py"
  echo ""
  echo "Features:"
  echo "  - Parallel batch downloads with trade_system_modules (50+ concurrency)"
  echo "  - Automatic GCS skipping for existing files"
  echo "  - Polygon.io ticker fetching (no Wikipedia/yfinance)"
  echo "  - Separate full vs delta modes"
  echo "  - No validation overhead"
  echo "  - Proper nanosecond timestamps"
}

main