"""
FastAPI wrapper around the Polymarket bot detectors.
"""

import asyncio
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from polymarket_bot import (
    fetch_active_markets,
    detect_overpriced_books,
    detect_monotonic_violations,
    detect_wide_spreads,
)

app = FastAPI(title="Polymarket Monitor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_executor = ThreadPoolExecutor(max_workers=4)


def _run_scan(check_spreads: bool) -> dict:
    markets = fetch_active_markets()
    signals = []
    signals.extend(detect_overpriced_books(markets))
    signals.extend(detect_monotonic_violations(markets))
    if check_spreads:
        signals.extend(detect_wide_spreads(markets))

    top_markets = sorted(markets, key=lambda m: m.volume, reverse=True)[:10]
    top_payload = [
        {
            "id": m.id,
            "question": m.question,
            "volume": round(m.volume, 2),
            "yes_price": m.yes_prices[0] if m.yes_prices else None,
            "outcomes": list(zip(m.outcomes, m.yes_prices)) if m.yes_prices else [],
        }
        for m in top_markets
    ]

    return {
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "market_count": len(markets),
        "signal_count": len(signals),
        "signals": [
            {
                "kind": s.kind,
                "severity": s.severity,
                "market_id": s.market_id,
                "question": s.question,
                "detail": s.detail,
            }
            for s in signals
        ],
        "top_markets": top_payload,
    }


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/scan")
async def scan(spreads: bool = Query(True, description="Include CLOB spread checks")):
    """Run all detectors and return trading signals."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run_scan, spreads)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}
