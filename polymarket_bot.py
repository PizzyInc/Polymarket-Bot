"""
Polymarket monitoring bot — detects price inefficiencies and emits trading signals.
"""

import argparse
import time
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

SPREAD_THRESHOLD   = 0.08   # flag spreads wider than 8 cents
OVERBOOK_THRESHOLD = 1.02   # sum of YES prices above this = overpriced book


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Market:
    id: str
    question: str
    outcomes: list[str]
    yes_prices: list[float]         # parallel to outcomes
    end_date: Optional[str]
    volume: float
    clob_token_ids: list[str] = field(default_factory=list)


@dataclass
class Signal:
    kind: str           # "OVERBOOK" | "MONOTONIC" | "SPREAD"
    market_id: str
    question: str
    detail: str
    severity: str       # "HIGH" | "MEDIUM" | "LOW"


# ---------------------------------------------------------------------------
# Gamma API helpers
# ---------------------------------------------------------------------------

def fetch_active_markets(limit: int = 200) -> list[Market]:
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume",
        "ascending": "false",
    }
    resp = requests.get(f"{GAMMA_API}/markets", params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    markets: list[Market] = []
    for m in raw:
        try:
            outcomes   = m.get("outcomes", [])
            out_prices = m.get("outcomePrices", [])
            if isinstance(outcomes, str):
                import json
                outcomes   = json.loads(outcomes)
                out_prices = json.loads(out_prices) if out_prices else []

            yes_prices = [float(p) for p in out_prices]
            clob_ids   = m.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                import json
                clob_ids = json.loads(clob_ids)

            markets.append(Market(
                id=str(m["id"]),
                question=m.get("question", ""),
                outcomes=outcomes,
                yes_prices=yes_prices,
                end_date=m.get("endDate"),
                volume=float(m.get("volume", 0) or 0),
                clob_token_ids=clob_ids,
            ))
        except (KeyError, ValueError, TypeError):
            continue

    return markets


# ---------------------------------------------------------------------------
# CLOB spread helper
# ---------------------------------------------------------------------------

def fetch_spread(token_id: str) -> Optional[float]:
    """Return best-ask minus best-bid for a CLOB token, or None on error."""
    try:
        resp = requests.get(
            f"{CLOB_API}/book",
            params={"token_id": token_id},
            timeout=10,
        )
        resp.raise_for_status()
        book = resp.json()
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if not bids or not asks:
            return None
        best_bid = max(float(b["price"]) for b in bids)
        best_ask = min(float(a["price"]) for a in asks)
        return round(best_ask - best_bid, 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Signal detectors
# ---------------------------------------------------------------------------

def detect_overpriced_books(markets: list[Market]) -> list[Signal]:
    signals: list[Signal] = []
    for m in markets:
        if len(m.yes_prices) < 2:
            continue
        book_sum = sum(m.yes_prices)
        if book_sum > OVERBOOK_THRESHOLD:
            excess = round(book_sum - 1.0, 4)
            signals.append(Signal(
                kind="OVERBOOK",
                market_id=m.id,
                question=m.question,
                detail=f"YES sum = {book_sum:.4f} (+{excess:.4f} over fair value)",
                severity="HIGH" if excess > 0.05 else "MEDIUM",
            ))
    return signals


def detect_monotonic_violations(markets: list[Market]) -> list[Signal]:
    """
    For related markets sharing a question prefix and sequential end dates,
    YES prices should be monotonically non-increasing as end dates extend
    (nearer resolution ≈ more certainty). Flag inversions.
    """
    signals: list[Signal] = []

    # Group by question prefix (first 60 chars) to find series
    from collections import defaultdict
    groups: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        if m.end_date:
            key = m.question[:60].strip()
            groups[key].append(m)

    for key, group in groups.items():
        if len(group) < 2:
            continue
        try:
            sorted_group = sorted(group, key=lambda m: m.end_date)
        except TypeError:
            continue

        for i in range(1, len(sorted_group)):
            prev, curr = sorted_group[i - 1], sorted_group[i]
            if not prev.yes_prices or not curr.yes_prices:
                continue
            prev_yes = prev.yes_prices[0]
            curr_yes = curr.yes_prices[0]
            # Longer-dated should be <= shorter-dated for a "will happen by X" market
            if curr_yes > prev_yes + 0.03:   # 3-cent tolerance
                signals.append(Signal(
                    kind="MONOTONIC",
                    market_id=curr.id,
                    question=curr.question,
                    detail=(
                        f"Price inversion: {prev.end_date[:10]} @ {prev_yes:.3f} "
                        f"< {curr.end_date[:10]} @ {curr_yes:.3f} "
                        f"(delta {curr_yes - prev_yes:+.3f})"
                    ),
                    severity="MEDIUM",
                ))

    return signals


def detect_wide_spreads(markets: list[Market], max_checks: int = 40) -> list[Signal]:
    """Check CLOB order books for markets with high volume and wide spreads."""
    signals: list[Signal] = []
    # Prioritise high-volume markets to limit API calls
    candidates = sorted(markets, key=lambda m: m.volume, reverse=True)[:max_checks]

    for m in candidates:
        if not m.clob_token_ids:
            continue
        token_id = m.clob_token_ids[0]
        spread = fetch_spread(token_id)
        if spread is not None and spread > SPREAD_THRESHOLD:
            signals.append(Signal(
                kind="SPREAD",
                market_id=m.id,
                question=m.question,
                detail=f"Bid-ask spread = {spread:.4f} (threshold {SPREAD_THRESHOLD})",
                severity="HIGH" if spread > 0.15 else "LOW",
            ))

    return signals


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SEVERITY_COLOR = {"HIGH": "\033[91m", "MEDIUM": "\033[93m", "LOW": "\033[96m"}
RESET = "\033[0m"

KIND_LABEL = {
    "OVERBOOK":  "OVERPRICED BOOK",
    "MONOTONIC": "MONOTONIC VIOLATION",
    "SPREAD":    "WIDE SPREAD",
}


def print_signals(signals: list[Signal]) -> None:
    if not signals:
        print("  (no signals detected)")
        return
    for s in signals:
        color  = SEVERITY_COLOR.get(s.severity, "")
        label  = KIND_LABEL.get(s.kind, s.kind)
        q_short = s.question[:80] + ("..." if len(s.question) > 80 else "")
        print(f"{color}[{s.severity:<6}] {label}{RESET}")
        print(f"         Market : {s.market_id}")
        print(f"         Q      : {q_short}")
        print(f"         Detail : {s.detail}")
        print()


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------

def run_scan(check_spreads: bool = True) -> list[Signal]:
    print(f"\n{'-'*60}")
    print(f"  Scan started  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-'*60}")

    print("  Fetching active markets from Gamma API...")
    markets = fetch_active_markets()
    print(f"  {len(markets)} markets loaded.\n")

    all_signals: list[Signal] = []

    ob_signals = detect_overpriced_books(markets)
    print(f"[OVERBOOK]   {len(ob_signals)} signal(s)")
    all_signals.extend(ob_signals)

    mono_signals = detect_monotonic_violations(markets)
    print(f"[MONOTONIC]  {len(mono_signals)} signal(s)")
    all_signals.extend(mono_signals)

    if check_spreads:
        print("[SPREAD]     checking top-volume CLOB books...")
        spread_signals = detect_wide_spreads(markets)
        print(f"[SPREAD]     {len(spread_signals)} signal(s)")
        all_signals.extend(spread_signals)

    print()
    print_signals(all_signals)
    return all_signals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket inefficiency monitor")
    parser.add_argument(
        "--watch", action="store_true",
        help="Continuously scan every 60 seconds",
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Refresh interval in seconds when --watch is active (default: 60)",
    )
    parser.add_argument(
        "--no-spreads", action="store_true",
        help="Skip CLOB spread checks (faster, fewer API calls)",
    )
    args = parser.parse_args()

    check_spreads = not args.no_spreads

    if args.watch:
        print(f"Watch mode active — scanning every {args.interval}s. Ctrl+C to stop.")
        try:
            while True:
                run_scan(check_spreads=check_spreads)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            sys.exit(0)
    else:
        run_scan(check_spreads=check_spreads)


if __name__ == "__main__":
    main()
