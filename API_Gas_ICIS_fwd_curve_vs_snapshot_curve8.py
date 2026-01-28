#!/usr/bin/env python3
"""
Streamlit app: ICIS Forward Gas Curves — streaming trades expansion + trade-only MTM mode + examples.

This updated version:
- Shows the raw PriceAssessments parsed table after "Build PriceAssessment curve".
- Joins trade period information (from trades' included period objects) to the trade-level MTM
  and uses a period label (relativePeriodCode or assessment mapping) in MTM illustration tables
  instead of the raw trade_id for clarity.
- Fixes Streamlit duplicate download_button element-id errors by assigning explicit unique keys
  to all download_button widgets.

Save as:
  API_Gas_curve_full_ICIS_streaming_with_tradeonly_and_examples.py

Run:
  pip install streamlit pandas requests plotly holidays
  streamlit run API_Gas_curve_full_ICIS_streaming_with_tradeonly_and_examples.py
"""
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import logging
import math
import zoneinfo
import csv
import tempfile
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional holidays lib
try:
    import holidays  # type: ignore
    _HAS_HOLIDAYS = True
except Exception:
    _HAS_HOLIDAYS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration / API endpoints
# ---------------------------
BASE_API = "https://api.icis.com/energy/v1"
SPEC_PRICEASSESSMENTS_ENDPOINT = urljoin(BASE_API + "/", "priceAssessments/specifications")
PRICE_ASSESSMENTS_ENDPOINT = urljoin(BASE_API + "/", "priceAssessments")
SPEC_TRADES_ENDPOINT = urljoin(BASE_API + "/", "trades/specifications")
TRADES_ENDPOINT = urljoin(BASE_API + "/", "trades")
TOKEN_URL = "https://api.icis.com/connect/token"
PAGE_SIZE = 1000

PRIORITY_ORDER = {
    "DA": 1, "DAY-AHEAD": 1,
    "WEEKEND": 2, "WDNW": 3, "BOM": 4,
    "MONTH": 5, "MONTHLY": 5,
    "Q": 6, "QUARTER": 6,
    "SEASON": 7, "WINTER": 7, "SUMMER": 7,
    "YEAR": 8, "GAS YEAR": 8,
}

# Gas-day start hours by country/hub timezone (local hour)
GAS_DAY_START_OVERRIDES = {"GB": 5, "UA": 5}  # local hour
DEFAULT_GAS_DAY_START = 6

# Map hub/country codes to IANA timezones (used for timestamp->gas-day mapping)
HUB_TIMEZONES = {
    "GB": "Europe/London",
    "NL": "Europe/Amsterdam",
    "DE": "Europe/Berlin",
    "FR": "Europe/Paris",
    "UA": "Europe/Kyiv",
}

# ---------------------------
# Small helpers
# ---------------------------
def make_session_with_retries(total_retries=4, backoff_factor=0.4, status_forcelist=(429, 500, 502, 503, 504)):
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["HEAD","GET","OPTIONS","POST","GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def get_token_and_expiry(client_id: str, client_secret: str) -> Dict[str, Any]:
    sess = make_session_with_retries()
    resp = sess.post(TOKEN_URL, data={"grant_type":"client_credentials"}, auth=(client_id, client_secret), timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"Token request failed: {resp.status_code} {resp.text[:1000]}")
    j = resp.json()
    token = j.get("access_token")
    expires_in = j.get("expires_in")
    if not token:
        raise RuntimeError(f"No access_token in token response: {j}")
    expires_at = None
    if isinstance(expires_in, (int,float)):
        expires_at = datetime.utcnow() + timedelta(seconds=int(expires_in))
    return {"access_token": token, "expires_at": expires_at, "raw": j}

def safe_parse_json(resp: requests.Response) -> Optional[Dict[str, Any]]:
    if resp.status_code == 204 or not resp.text:
        return None
    try:
        return resp.json()
    except ValueError:
        logger.exception("Failed to parse JSON")
        raise

def choose_df(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return first non-empty DataFrame, else empty DataFrame."""
    if isinstance(a, pd.DataFrame) and not a.empty:
        return a
    if isinstance(b, pd.DataFrame) and not b.empty:
        return b
    return pd.DataFrame()

# ---------------------------
# API fetchers (paginated)
# ---------------------------
def fetch_spec_price_assessments(token: str) -> List[Dict[str, Any]]:
    sess = make_session_with_retries()
    sess.headers.update({"Accept":"application/json","Authorization":f"Bearer {token}"})
    resp = sess.get(SPEC_PRICEASSESSMENTS_ENDPOINT, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])

def fetch_price_assessments(token: str, specification_id: str, start_str: str, end_str: str) -> Dict[str, Any]:
    sess = make_session_with_retries()
    sess.headers.update({"Accept":"application/json","Authorization":f"Bearer {token}"})
    params = {
        "filter[specificationId]": specification_id,
        "filter[createdForDateStart]": start_str,
        "filter[createdForDateEnd]": end_str,
        "page[size]": PAGE_SIZE,
    }
    url = PRICE_ASSESSMENTS_ENDPOINT
    all_data = []
    all_included = []
    while url:
        resp = sess.get(url, params=params if params else None, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"PriceAssessments request failed: {resp.status_code} {resp.text[:1000]}")
        page = safe_parse_json(resp)
        if not page:
            break
        all_data.extend(page.get("data", []) or [])
        all_included.extend(page.get("included", []) or [])
        next_link = page.get("links", {}).get("next")
        if not next_link:
            break
        if urlparse(next_link).netloc:
            url = next_link
        else:
            url = urljoin(BASE_API + "/", next_link.lstrip("/"))
        params = None
    return {"data": all_data, "included": all_included}

def fetch_trade_specifications(token: str) -> List[Dict[str, Any]]:
    sess = make_session_with_retries()
    sess.headers.update({"Accept":"application/json","Authorization":f"Bearer {token}"})
    resp = sess.get(SPEC_TRADES_ENDPOINT, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])

def fetch_trades(token: str, specification_ids: List[str], start_str: str, end_str: str) -> Dict[str, Any]:
    sess = make_session_with_retries()
    sess.headers.update({"Accept":"application/json","Authorization":f"Bearer {token}"})
    combined_data = []
    combined_included = []
    for spec_id in specification_ids:
        params = {
            "filter[specificationId]": spec_id,
            "filter[createdForDateStart]": start_str,
            "filter[createdForDateEnd]": end_str,
            "page[size]": PAGE_SIZE,
        }
        url = TRADES_ENDPOINT
        while url:
            resp = sess.get(url, params=params if params else None, timeout=60)
            if resp.status_code >= 400:
                raise RuntimeError(f"Trades request failed: {resp.status_code} {resp.text[:1000]}")
            page = safe_parse_json(resp)
            if not page:
                break
            combined_data.extend(page.get("data", []) or [])
            combined_included.extend(page.get("included", []) or [])
            next_link = page.get("links", {}).get("next")
            if not next_link:
                break
            if urlparse(next_link).netloc:
                url = next_link
            else:
                url = urljoin(BASE_API + "/", next_link.lstrip("/"))
            params = None
    return {"data": combined_data, "included": combined_included}

# ---------------------------
# Parsing helpers
# ---------------------------
def parse_assessments_with_periods(combined: Dict[str, Any]) -> pd.DataFrame:
    data = combined.get("data", []) or []
    included = combined.get("included", []) or []
    included_index = {(it.get("type"), it.get("id")): it for it in included}
    rows = []
    for item in data:
        atts = item.get("attributes") or {}
        rels = item.get("relationships") or {}
        period_ref = rels.get("period", {}).get("data") or {}
        ptype = period_ref.get("type")
        pid = period_ref.get("id")
        period_obj = included_index.get((ptype, pid), {}) if ptype and pid else {}
        period_attrs = period_obj.get("attributes") or {}
        rows.append({
            "assessment_id": item.get("id"),
            "midpoint": atts.get("midpoint"),
            "createdForDate": atts.get("createdForDate"),
            "period_id": pid,
            "period_startDate": period_attrs.get("startDate"),
            "period_endDate": period_attrs.get("endDate"),
            "period_defaultPeriodLabel": period_attrs.get("defaultPeriodLabel"),
            "period_relativePeriodCode": period_attrs.get("relativePeriodCode"),
            "period_relativePeriodLabel": period_attrs.get("relativePeriodLabel"),
            "period_displayOrder": period_attrs.get("displayOrder"),
        })
    return pd.DataFrame(rows)

def parse_trades_with_periods(combined: Dict[str, Any]) -> pd.DataFrame:
    data = combined.get("data", []) or []
    included = combined.get("included", []) or []
    included_index = {(it.get("type"), it.get("id")): it for it in included}
    rows = []
    for item in data:
        atts = item.get("attributes") or {}
        rels = item.get("relationships") or {}
        period_ref = rels.get("period", {}).get("data") or {}
        pid = period_ref.get("id")
        period_obj = included_index.get(("period", pid), {}) if pid else {}
        period_attrs = period_obj.get("attributes") or {}
        ts_ref = rels.get("tradeSpecification", {}).get("data") or {}
        ts_id = ts_ref.get("id")
        ts_obj = included_index.get(("tradeSpecification", ts_id), {}) if ts_id else {}
        ts_attrs = ts_obj.get("attributes") or {}
        rows.append({
            "trade_id": item.get("id"),
            "createdForDate": atts.get("createdForDate"),
            "modifiedDateTime": atts.get("modifiedDateTime"),
            "tradeDateTime": atts.get("tradeDateTime"),
            "price": atts.get("price"),
            "volume": atts.get("volume"),
            "totalVolume": atts.get("totalVolume"),
            "tradeType": atts.get("tradeType"),
            "isIndexQualified": atts.get("isIndexQualified"),
            "tradeSide": atts.get("tradeSide"),
            "marketDeliveryProfileName": atts.get("marketDeliveryProfileName"),
            "period_id": pid,
            "period_referenceDate": period_attrs.get("referenceDate"),
            "period_defaultPeriodLabel": period_attrs.get("defaultPeriodLabel"),
            "period_startDate": period_attrs.get("startDate"),
            "period_endDate": period_attrs.get("endDate"),
            "period_displayOrder": period_attrs.get("displayOrder"),
            "period_absolutePeriodCode": period_attrs.get("absolutePeriodCode"),
            "period_absolutePeriodLabel": period_attrs.get("absolutePeriodLabel"),
            "period_relativePeriodCode": period_attrs.get("relativePeriodCode"),
            "period_relativePeriodLabel": period_attrs.get("relativePeriodLabel"),
            "period_defaultPeriodCode": period_attrs.get("defaultPeriodCode"),
            "period_periodType": period_attrs.get("periodType"),
            "tradeSpecification_id": ts_id,
            "tradeSpecification_name": ts_attrs.get("name"),
            "tradeSpecification_code": ts_attrs.get("code"),
        })
    return pd.DataFrame(rows)

# ---------------------------
# Utilities: timezone & gas-day mapping
# ---------------------------
def get_timezone_for_country(country_code: str) -> str:
    return HUB_TIMEZONES.get(country_code.upper(), "Europe/Berlin")

def get_gas_day_start_hour(country_code: Optional[str]) -> int:
    if not country_code:
        return DEFAULT_GAS_DAY_START
    return GAS_DAY_START_OVERRIDES.get(country_code.upper(), DEFAULT_GAS_DAY_START)

def timestamp_to_gas_day(ts: str, tz_name: str, gas_day_start_hour: int) -> Optional[date]:
    if pd.isna(ts) or ts is None:
        return None
    try:
        dt = pd.to_datetime(ts)
        try:
            tz = zoneinfo.ZoneInfo(tz_name)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(tz)
            else:
                dt = dt.astimezone(tz)
        except Exception:
            pass
        local_date = dt.date()
        local_time = dt.time()
        if local_time.hour < gas_day_start_hour:
            gas_day = local_date - timedelta(days=1)
        else:
            gas_day = local_date
        return gas_day
    except Exception:
        return None

# ---------------------------
# Curve construction helpers
# ---------------------------
def get_priority_from_code(code: Optional[str]) -> int:
    if not code:
        return 999
    c = str(code).upper()
    for k, v in PRIORITY_ORDER.items():
        if k in c:
            return v
    if "DAY" in c and "AHEAD" in c:
        return 1
    return 999

def build_daily_curve_allocation(df_assess: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["date", "market_price", "source_assessment_id", "source_period_id", "source_label", "priority"]
    if df_assess is None or not isinstance(df_assess, pd.DataFrame) or df_assess.empty:
        return pd.DataFrame(columns=out_cols)
    df = df_assess.copy()
    for c in ["period_startDate", "period_endDate", "midpoint"]:
        if c not in df.columns:
            df[c] = pd.NA
    def safe_date(x):
        try:
            return pd.to_datetime(x).date()
        except Exception:
            return pd.NaT
    df["period_start_date"] = df["period_startDate"].apply(safe_date)
    df["period_end_date"] = df["period_endDate"].apply(safe_date)
    df["midpoint_num"] = pd.to_numeric(df["midpoint"], errors="coerce")
    df = df.dropna(subset=["period_start_date", "period_end_date", "midpoint_num"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=out_cols)
    overall_start = df["period_start_date"].min()
    overall_end = df["period_end_date"].max()
    all_days = pd.date_range(overall_start, overall_end, freq="D").date
    out = pd.DataFrame({"date": all_days})
    out["market_price"] = pd.NA
    out["priority"] = pd.NA
    out["source_assessment_id"] = pd.NA
    out["source_period_id"] = pd.NA
    out["source_label"] = pd.NA
    df["priority"] = df.get("period_relativePeriodCode").apply(get_priority_from_code) if "period_relativePeriodCode" in df.columns else 999
    df["period_len"] = (pd.to_datetime(df["period_end_date"]) - pd.to_datetime(df["period_start_date"])).dt.days + 1
    df["period_displayOrder_num"] = pd.to_numeric(df.get("period_displayOrder", pd.Series([pd.NA] * len(df))), errors="coerce").fillna(999999)
    df = df.sort_values(by=["priority", "period_displayOrder_num", "period_len"]).reset_index(drop=True)
    for _, r in df.iterrows():
        s = r["period_start_date"]; e = r["period_end_date"]
        mid = float(r["midpoint_num"])
        assessment_id = r.get("assessment_id")
        period_id = r.get("period_id")
        label = r.get("period_defaultPeriodLabel") or r.get("period_relativePeriodLabel") or r.get("period_relativePeriodCode")
        pr = int(r["priority"]) if not pd.isna(r["priority"]) else 999
        if pd.isna(s) or pd.isna(e):
            continue
        try:
            for d in pd.date_range(start=s, end=e, freq="D").date:
                idxs = out.index[out["date"] == d].tolist()
                if not idxs:
                    continue
                idx = idxs[0]
                existing_pr = out.at[idx, "priority"]
                if pd.isna(existing_pr) or int(existing_pr) > pr:
                    out.at[idx, "market_price"] = float(mid)
                    out.at[idx, "priority"] = pr
                    out.at[idx, "source_assessment_id"] = assessment_id
                    out.at[idx, "source_period_id"] = period_id
                    out.at[idx, "source_label"] = label
        except Exception as ex:
            logger.exception("Error applying assessment %s to dates %s..%s: %s", assessment_id, s, e, ex)
            continue
    out["market_price"] = pd.to_numeric(out["market_price"], errors="coerce")
    return out

def compute_period_prices_from_assessments(df_assess: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if df_assess is None or df_assess.empty:
        return pd.DataFrame(columns=["period_id", "period_price", "period_startDate", "period_endDate", "period_label", "n_assessments", "mean_midpoint"])
    df = df_assess.copy()
    df["midpoint_num"] = pd.to_numeric(df["midpoint"], errors="coerce")
    def agg_group(g):
        out = {}
        out["period_startDate"] = g["period_startDate"].dropna().iloc[0] if g["period_startDate"].notna().any() else None
        out["period_endDate"] = g["period_endDate"].dropna().iloc[0] if g["period_endDate"].notna().any() else None
        out["period_label"] = (g["period_defaultPeriodLabel"].dropna().iloc[0] if "period_defaultPeriodLabel" in g and g["period_defaultPeriodLabel"].notna().any() else 
                               (g.get("period_relativePeriodLabel", pd.Series()).dropna().iloc[0] if "period_relativePeriodLabel" in g and g["period_relativePeriodLabel"].notna().any() else None))
        out["n_assessments"] = int(len(g))
        out["mean_midpoint"] = float(g["midpoint_num"].mean(skipna=True)) if not g["midpoint_num"].dropna().empty else np.nan
        if method == "median":
            out["period_price"] = float(g["midpoint_num"].median(skipna=True))
        elif method == "latest":
            if "createdForDate" in g.columns:
                sorted_g = g.sort_values("createdForDate", ascending=False)
                out["period_price"] = float(pd.to_numeric(sorted_g["midpoint"].iloc[0], errors="coerce"))
            else:
                out["period_price"] = out["mean_midpoint"]
        else:
            out["period_price"] = out["mean_midpoint"]
        return pd.Series(out)
    grouped = df.groupby("period_id").apply(agg_group).reset_index()
    return grouped[["period_id", "period_price", "period_startDate", "period_endDate", "period_label", "n_assessments", "mean_midpoint"]]

def compute_period_prices_from_trades(trades_df: pd.DataFrame, method: str = "VWAP", trim_pct: float = 0.1) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    def est_daily_vol(row):
        try:
            if pd.notna(row.get("volume")):
                return float(row["volume"])
            tv = row.get("totalVolume")
            s = pd.to_datetime(row.get("period_startDate"))
            e = pd.to_datetime(row.get("period_endDate"))
            n = (e.date() - s.date()).days + 1 if s and e else 1
            return float(tv) / n if tv is not None and n > 0 else np.nan
        except Exception:
            return np.nan
    df["daily_vol_est"] = df.apply(est_daily_vol, axis=1)
    df["price_num"] = pd.to_numeric(df["price"], errors="coerce")
    rows = []
    for pid, g in df.groupby("period_id", dropna=False):
        priced = g[~g["price_num"].isna()].copy()
        if priced.empty:
            continue
        gross_vol = float(priced["daily_vol_est"].sum(skipna=True))
        if method == "VWAP":
            denom = priced["daily_vol_est"].sum(skipna=True)
            if denom > 0:
                p = float((priced["price_num"] * priced["daily_vol_est"]).sum(skipna=True) / denom)
            else:
                p = float(priced["price_num"].mean(skipna=True))
        elif method == "median":
            p = float(priced["price_num"].median(skipna=True))
        elif method == "trimmed":
            low = priced["price_num"].quantile(trim_pct)
            high = priced["price_num"].quantile(1 - trim_pct)
            trimmed = priced[(priced["price_num"] >= low) & (priced["price_num"] <= high)]
            p = float(trimmed["price_num"].mean(skipna=True)) if not trimmed.empty else float(priced["price_num"].mean(skipna=True))
        else:
            p = float(priced["price_num"].mean(skipna=True))
        rep = g.iloc[0]
        rows.append({
            "period_id": pid,
            "period_price": p,
            "period_startDate": rep.get("period_startDate"),
            "period_endDate": rep.get("period_endDate"),
            "period_label": rep.get("period_relativePeriodLabel") or rep.get("period_defaultPeriodLabel"),
            "n_trades": int(len(priced)),
            "gross_volume": gross_vol,
        })
    return pd.DataFrame(rows)

def expand_period_prices_to_daily(period_prices_df: pd.DataFrame, prefer_shortest_if_overlap: bool = True) -> pd.DataFrame:
    if period_prices_df is None or period_prices_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in period_prices_df.iterrows():
        try:
            s = pd.to_datetime(r["period_startDate"]).date() if "period_startDate" in r and pd.notna(r["period_startDate"]) else None
            e = pd.to_datetime(r["period_endDate"]).date() if "period_endDate" in r and pd.notna(r["period_endDate"]) else None
        except Exception:
            s = None; e = None
        if s is None or e is None:
            continue
        for d in pd.date_range(start=s, end=e, freq="D").date:
            rows.append({
                "date": d,
                "period_price": r["period_price"],
                "period_id": r.get("period_id"),
                "period_startDate": r.get("period_startDate"),
                "period_endDate": r.get("period_endDate"),
                "period_label": r.get("period_label"),
                "period_len": (pd.to_datetime(r["period_endDate"]) - pd.to_datetime(r["period_startDate"])).days + 1 if r.get("period_startDate") and r.get("period_endDate") else 9999
            })
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    if prefer_shortest_if_overlap:
        daily = daily.sort_values(["date", "period_len"]).drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily[["date", "period_price", "period_id", "period_startDate", "period_endDate", "period_label"]]

# ---------------------------
# Arbitrage-free monthly derivation
# ---------------------------
def enforce_arbitrage_free_monthly(period_prices_df: pd.DataFrame) -> pd.DataFrame:
    if period_prices_df is None or period_prices_df.empty:
        return pd.DataFrame(columns=["year","month","period_price","source_period_id"])
    df = period_prices_df.copy()
    df["start_dt"] = pd.to_datetime(df["period_startDate"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["period_endDate"], errors="coerce")
    df["len_days"] = (df["end_dt"] - df["start_dt"]).dt.days + 1
    monthly = df[(df["len_days"] >= 25) & (df["len_days"] <= 35)].copy()
    quarter = df[(df["len_days"] >= 80) & (df["len_days"] <= 100)].copy()
    season = df[(df["len_days"] >= 160) & (df["len_days"] <= 200)].copy()
    year = df[(df["len_days"] >= 350)].copy()
    month_map: Dict[Tuple[int,int], Dict[str, Any]] = {}
    for _, r in monthly.iterrows():
        s = r["start_dt"]
        if pd.isna(s):
            continue
        month_map[(int(s.year), int(s.month))] = {"price": float(r["period_price"]), "source_period_id": r["period_id"]}
    def fill_from_broader(rows_df: pd.DataFrame, months_in_period: int):
        for _, r in rows_df.iterrows():
            s = r["start_dt"]
            if pd.isna(s):
                continue
            months = []
            for i in range(months_in_period):
                month_dt = (s + pd.DateOffset(months=i))
                months.append((int(month_dt.year), int(month_dt.month)))
            known_sum = 0.0
            missing = []
            for ym in months:
                if ym in month_map:
                    known_sum += float(month_map[ym]["price"])
                else:
                    missing.append(ym)
            if len(missing) == 0:
                continue
            total_needed = months_in_period * float(r["period_price"])
            remaining = total_needed - known_sum
            per_missing = remaining / len(missing)
            for ym in missing:
                if ym not in month_map:
                    month_map[ym] = {"price": float(per_missing), "source_period_id": r["period_id"]}
    fill_from_broader(quarter, 3)
    fill_from_broader(season, 6)
    fill_from_broader(year, 12)
    rows = []
    for (y,m), info in sorted(month_map.items()):
        rows.append({"year": y, "month": m, "period_price": info["price"], "source_period_id": info.get("source_period_id")})
    out = pd.DataFrame(rows)
    return out

# ---------------------------
# Holiday & working-day helpers
# ---------------------------
def get_holiday_calendar(country_code: str, years: List[int]):
    if not _HAS_HOLIDAYS:
        raise RuntimeError("python-holidays not installed. Install with `pip install holidays`.")
    try:
        cal = holidays.CountryHoliday(country_code, years=years)
        return cal
    except Exception as e:
        raise RuntimeError(f"Failed to create holiday calendar for country '{country_code}': {e}")

def is_working_day(dt: date, country_code: str) -> bool:
    if not isinstance(dt, (date, datetime)):
        dt = pd.to_datetime(dt).date()
    if not _HAS_HOLIDAYS:
        return dt.weekday() < 5
    try:
        cal = get_holiday_calendar(country_code, [dt.year])
        return (dt.weekday() < 5) and (dt not in cal)
    except Exception:
        return dt.weekday() < 5

def next_working_gas_day(report_date: date, country_code: str) -> date:
    d = report_date + timedelta(days=1)
    while not is_working_day(d, country_code):
        d += timedelta(days=1)
    return d

def next_weekend(report_date: date, country_code: str) -> List[date]:
    d = report_date
    days_ahead = (5 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    saturday = d + timedelta(days=days_ahead)
    weekend_days = [saturday, saturday + timedelta(days=1)]
    monday = saturday + timedelta(days=2)
    if not is_working_day(monday, country_code):
        weekend_days.append(monday)
    return [dd for dd in weekend_days if isinstance(dd, (date, datetime))]

def next_week_working_days(report_date: date, country_code: str) -> List[date]:
    d = report_date
    days_ahead = (7 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_monday = d + timedelta(days=days_ahead)
    workdays = []
    for i in range(5):
        day = next_monday + timedelta(days=i)
        if is_working_day(day, country_code):
            workdays.append(day)
    return workdays

# ---------------------------
# WDNW & BOM derivation + weighting
# ---------------------------
def derive_wdnw_days_and_price(report_date: date, alloc_curve: pd.DataFrame, assess_daily: pd.DataFrame, trades_daily: pd.DataFrame, country_code: str, wdnw_mode: str = "Equal", da_weight: float = 0.5) -> Tuple[List[date], Optional[float]]:
    wdnw_days = next_week_working_days(report_date, country_code)
    if not wdnw_days:
        return [], None
    day_prices = {}
    for d in wdnw_days:
        p = None
        if alloc_curve is not None and not alloc_curve.empty:
            row = alloc_curve[alloc_curve["date"] == pd.to_datetime(d).date()]
            if not row.empty and pd.notna(row.iloc[0]["market_price"]):
                p = float(row.iloc[0]["market_price"])
        if p is None and assess_daily is not None and not assess_daily.empty:
            row = assess_daily[assess_daily["date"] == pd.to_datetime(d).date()]
            if not row.empty and pd.notna(row.iloc[0]["market_price"]):
                p = float(row.iloc[0]["market_price"])
        if p is None and trades_daily is not None and not trades_daily.empty:
            row = trades_daily[trades_daily["date"] == pd.to_datetime(d).date()]
            if not row.empty and pd.notna(row.iloc[0]["market_price"]):
                p = float(row.iloc[0]["market_price"])
        day_prices[d] = p
    priced_days = [d for d, p in day_prices.items() if p is not None]
    if not priced_days:
        return wdnw_days, None
    if wdnw_mode == "Equal":
        avg_price = float(np.mean([day_prices[d] for d in priced_days]))
        return wdnw_days, avg_price
    da = next_working_gas_day(report_date, country_code)
    da = pd.to_datetime(da).date()
    if da not in priced_days:
        avg_price = float(np.mean([day_prices[d] for d in priced_days]))
        return wdnw_days, avg_price
    da_w = float(max(0.0, min(1.0, da_weight)))
    other_days = [d for d in priced_days if d != da]
    n_other = len(other_days)
    if n_other == 0:
        return wdnw_days, float(day_prices[da])
    other_total = 1.0 - da_w
    per_other = other_total / n_other
    weighted = da_w * day_prices[da] + sum(per_other * day_prices[d] for d in other_days)
    return wdnw_days, float(weighted)

def derive_bom_days_and_price(report_date: date, alloc_curve: pd.DataFrame, assess_daily: pd.DataFrame, trades_daily: pd.DataFrame, country_code: str) -> Tuple[List[date], Optional[float]]:
    da = next_working_gas_day(report_date, country_code)
    wkend = next_weekend(report_date, country_code)
    wkend_start = min(wkend) if wkend else da
    earlier = min(pd.to_datetime(da).date(), pd.to_datetime(wkend_start).date())
    bom_start = earlier + timedelta(days=1)
    if bom_start.month != report_date.month:
        bom_start = (report_date.replace(day=1) + pd.DateOffset(months=1)).date()
    month_end = (pd.Timestamp(bom_start).replace(day=1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date()
    days = pd.date_range(start=bom_start, end=month_end, freq="D").date
    prices = []
    for d in days:
        p = None
        if alloc_curve is not None and not alloc_curve.empty:
            row = alloc_curve[alloc_curve["date"] == pd.to_datetime(d).date()]
            if not row.empty and pd.notna(row.iloc[0]["market_price"]):
                p = float(row.iloc[0]["market_price"])
        if p is None and assess_daily is not None and not assess_daily.empty:
            row = assess_daily[assess_daily["date"] == pd.to_datetime(d).date()]
            if not row.empty and pd.notna(row.iloc[0]["market_price"]):
                p = float(row.iloc[0]["market_price"])
        if p is None and trades_daily is not None and not trades_daily.empty:
            row = trades_daily[trades_daily["date"] == pd.to_datetime(d).date()]
            if not row.empty and pd.notna(row.iloc[0]["market_price"]):
                p = float(row.iloc[0]["market_price"])
        if p is not None:
            prices.append(p)
    if not prices:
        return list(days), None
    return list(days), float(np.mean(prices))

# ---------------------------
# Shaping helpers (as earlier)
# ---------------------------
def compute_bom_weekday_factors_from_history(daily_price_history: pd.DataFrame, reference_month: int, window_years: int = 5,
                                             front_month_mode: str = "calendar", period_prices_hist: Optional[pd.DataFrame] = None) -> Dict[int, float]:
    if daily_price_history is None or daily_price_history.empty:
        return {i: 1.0 for i in range(7)}
    df = daily_price_history.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["price"])
    now = datetime.utcnow().date()
    years = [now.year - i for i in range(1, window_years + 1)]
    per_year_weekday_avgs = []
    front_month_avgs = []
    for y in years:
        try:
            end = pd.Timestamp(year=y, month=reference_month, day=1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            start = end - pd.DateOffset(months=11)
            window_df = df[(df["date_dt"] >= start) & (df["date_dt"] <= end)]
            if window_df.empty:
                continue
            window_df["dow"] = window_df["date_dt"].dt.weekday
            weekday_mean = window_df.groupby("dow")["price"].mean()
            per_year_weekday_avgs.append(weekday_mean)
            if front_month_mode == "calendar":
                fm_df = window_df[window_df["date_dt"].dt.month == 1]
                fm_avg = float(fm_df["price"].mean()) if not fm_df.empty else float(window_df["price"].mean())
            else:
                fm_avg = None
                if period_prices_hist is not None and not period_prices_hist.empty:
                    ph = period_prices_hist.copy()
                    ph["start_dt"] = pd.to_datetime(ph["period_startDate"], errors="coerce")
                    ph["len_days"] = (pd.to_datetime(ph["period_endDate"]) - pd.to_datetime(ph["period_startDate"])).dt.days + 1
                    candidates = ph[(ph["start_dt"] > end) & (ph["len_days"].between(25,35))].sort_values("start_dt")
                    if not candidates.empty:
                        front_start = candidates.iloc[0]["start_dt"]
                        fm_month = int(pd.to_datetime(front_start).month)
                        fm_df = window_df[window_df["date_dt"].dt.month == fm_month]
                        fm_avg = float(fm_df["price"].mean()) if not fm_df.empty else float(window_df["price"].mean())
                if fm_avg is None:
                    fm_avg = float(window_df["price"].mean())
            front_month_avgs.append(fm_avg)
        except Exception:
            continue
    if not per_year_weekday_avgs:
        return {i: 1.0 for i in range(7)}
    stacked = pd.concat(per_year_weekday_avgs, axis=1)
    mean_by_dow = stacked.mean(axis=1)
    fm_avg = float(np.mean(front_month_avgs)) if front_month_avgs else mean_by_dow.mean()
    factors = {int(d): float(mean_by_dow.get(d, 0.0) / fm_avg if fm_avg and fm_avg != 0 else 1.0) for d in range(7)}
    mean_factor = np.mean(list(factors.values()))
    if mean_factor and mean_factor != 0:
        factors = {d: (factors[d] / mean_factor) for d in factors}
    return factors

def compute_monthly_granularity_shaping_from_history(period_prices_assess: pd.DataFrame, window_years: int = 5) -> Dict[int, float]:
    if period_prices_assess is None or period_prices_assess.empty:
        return {m:1.0 for m in range(1,13)}
    df = period_prices_assess.copy()
    df["start_dt"] = pd.to_datetime(df["period_startDate"], errors="coerce")
    df = df.dropna(subset=["start_dt","period_price"])
    now = datetime.utcnow().date()
    years = [now.year - i for i in range(1, window_years+1)]
    month_vals = {m: [] for m in range(1,13)}
    for _, r in df.iterrows():
        s = r["start_dt"]; y = s.year; m = s.month
        if y in years:
            month_vals[m].append(float(r["period_price"]))
    month_avg = {m: (np.mean(v) if v else np.nan) for m,v in month_vals.items()}
    quarter_map = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
    quarter_avg = {}
    for q in [1,2,3,4]:
        months = [m for m,qq in quarter_map.items() if qq == q]
        vals = [month_avg[m] for m in months if not np.isnan(month_avg[m])]
        quarter_avg[q] = np.mean(vals) if vals else np.nan
    ratios = {}
    for m in range(1,13):
        q = quarter_map[m]
        if not np.isnan(month_avg.get(m, np.nan)) and not np.isnan(quarter_avg.get(q, np.nan)) and quarter_avg.get(q, np.nan) != 0:
            ratios[m] = month_avg[m] / quarter_avg[q]
        else:
            ratios[m] = 1.0
    meanv = np.mean(list(ratios.values()))
    factors = {m: ratios[m] / meanv for m in ratios}
    return factors

# ---------------------------
# Trade expansion helpers
# ---------------------------
def expand_trade_to_daily(trade_row: Dict[str, Any], assume_side: str = "BUY", hub_country: Optional[str] = "GB") -> pd.DataFrame:
    """
    Expand a single trade to daily rows. If period_startDate/endDate present, use them;
    otherwise map tradeDateTime to gas-day using hub timezone and gas-day start hour.
    """
    start_raw = trade_row.get("period_startDate")
    end_raw = trade_row.get("period_endDate")
    trade_dt_raw = trade_row.get("tradeDateTime")
    tz_name = get_timezone_for_country(hub_country) if hub_country else "Europe/Berlin"
    gas_day_start_hour = get_gas_day_start_hour(hub_country)
    try:
        if start_raw and end_raw:
            start = pd.to_datetime(start_raw).date()
            end = pd.to_datetime(end_raw).date()
        else:
            if trade_dt_raw:
                gd = timestamp_to_gas_day(trade_dt_raw, tz_name, gas_day_start_hour)
                start = gd; end = gd
            else:
                start = pd.to_datetime(trade_row.get("createdForDate") or datetime.utcnow().date()).date()
                end = start
    except Exception:
        start = pd.to_datetime(trade_row.get("createdForDate") or datetime.utcnow().date()).date()
        end = start
    days = pd.date_range(start=start, end=end, freq="D").date
    n_days = len(days) if len(days) > 0 else 1
    vol_attr = trade_row.get("volume")
    total_vol = trade_row.get("totalVolume")
    if vol_attr is not None:
        try:
            daily_vol = float(vol_attr)
        except Exception:
            daily_vol = 0.0
    else:
        if total_vol is not None:
            try:
                daily_vol = float(total_vol) / n_days
            except Exception:
                daily_vol = 0.0
        else:
            daily_vol = 0.0
    trade_side = trade_row.get("tradeSide") or assume_side
    trade_price = trade_row.get("price")
    rows = []
    period_fields = {k: trade_row.get(k) for k in trade_row.keys() if k.startswith("period_")}
    if "period_id" in trade_row:
        period_fields["period_id"] = trade_row.get("period_id")
    ts_fields = {
        "tradeSpecification_name": trade_row.get("tradeSpecification_name"),
        "tradeSpecification_code": trade_row.get("tradeSpecification_code"),
        "tradeSpecification_id": trade_row.get("tradeSpecification_id"),
    }
    for d in days:
        r = {
            "trade_id": trade_row.get("trade_id"),
            "date": d,
            "daily_volume": daily_vol,
            "trade_price": trade_price,
            "trade_side": (trade_side or assume_side),
        }
        r.update(period_fields)
        r.update(ts_fields)
        rows.append(r)
    return pd.DataFrame(rows)

# ---------------------------
# Compute insights (added here so it's available)
# ---------------------------
def compute_insights(merged: pd.DataFrame, trade_mtm_df: pd.DataFrame, trades_daily_df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["num_trades"] = int(trade_mtm_df.shape[0]) if trade_mtm_df is not None else 0
    out["portfolio_mtm"] = float(trade_mtm_df["trade_mtm"].sum()) if trade_mtm_df is not None and not trade_mtm_df.empty else 0.0
    out["positive_trades"] = int((trade_mtm_df["trade_mtm"] > 0).sum()) if trade_mtm_df is not None and not trade_mtm_df.empty else 0
    out["negative_trades"] = int((trade_mtm_df["trade_mtm"] < 0).sum()) if trade_mtm_df is not None and not trade_mtm_df.empty else 0
    out["top_trades"] = trade_mtm_df.sort_values("trade_mtm", ascending=False).head(10) if trade_mtm_df is not None and not trade_mtm_df.empty else pd.DataFrame()
    out["worst_trades"] = trade_mtm_df.sort_values("trade_mtm", ascending=True).head(10) if trade_mtm_df is not None and not trade_mtm_df.empty else pd.DataFrame()
    # daily contribution
    if merged is not None and not merged.empty:
        daily = merged.groupby("date", as_index=False)["mtm_day"].sum().rename(columns={"mtm_day": "daily_mtm"})
        out["top_dates"] = daily.sort_values("daily_mtm", ascending=False).head(10)
        out["bottom_dates"] = daily.sort_values("daily_mtm", ascending=True).head(10)
        out["missing_market_rows"] = int(merged["market_price"].isna().sum())
        out["missing_market_dates"] = merged[merged["market_price"].isna()]["date"].drop_duplicates().tolist()
    else:
        out["top_dates"] = pd.DataFrame()
        out["bottom_dates"] = pd.DataFrame()
        out["missing_market_rows"] = 0
        out["missing_market_dates"] = []
    # net volume by side
    if merged is not None and not merged.empty:
        dfv = merged.copy()
        dfv["side_sign"] = dfv["trade_side"].fillna("BUY").str.upper().map(lambda s: 1 if s == "BUY" else -1)
        dfv["signed_vol"] = dfv["daily_volume"].astype(float) * dfv["side_sign"]
        out["net_volume_by_side"] = dfv.groupby("trade_side", dropna=False)["daily_volume"].sum().to_dict()
        dfv["date_dt"] = pd.to_datetime(dfv["date"])
        dfv["month"] = dfv["date_dt"].dt.to_period("M").astype(str)
        dfv["market_price_filled"] = dfv["market_price"].fillna(0.0)
        out["exposure_by_month"] = dfv.groupby("month").apply(lambda g: float((g["market_price_filled"] * g["daily_volume"]).sum())).to_dict()
        tb = merged.copy()
        tb_nonnull = tb[~tb["market_price"].isna()].copy()
        if not tb_nonnull.empty:
            weighted = tb_nonnull.assign(weighted=tb_nonnull["market_price"] * tb_nonnull["daily_volume"]).groupby("trade_id").agg({"weighted": "sum", "daily_volume": "sum"})
            weighted["breakeven_market_price"] = weighted["weighted"] / weighted["daily_volume"]
            breakeven = weighted["breakeven_market_price"].reset_index()
            out["trade_breakeven"] = breakeven
        else:
            out["trade_breakeven"] = pd.DataFrame(columns=["trade_id", "breakeven_market_price"])
    else:
        out["net_volume_by_side"] = {}
        out["exposure_by_month"] = {}
        out["trade_breakeven"] = pd.DataFrame(columns=["trade_id", "breakeven_market_price"])
    return out

# ---------------------------
# Streaming merged + trade-only MTM processors
# ---------------------------
def process_trades_streaming_and_compute_mtm(trades_df: pd.DataFrame, valuation_curve: pd.DataFrame, hub_country: str = "GB",
                                             assume_side: str = "BUY", sample_rows: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame, float, str]:
    """
    Stream expand trades and compute MTM while writing merged daily rows to a temp CSV to avoid high memory.
    Returns (merged_sample_df, trade_mtm_df, portfolio_mtm, merged_csv_path)
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0, ""

    # prepare date -> price map
    price_map = {}
    if valuation_curve is not None and not valuation_curve.empty:
        vc = valuation_curve[["date","market_price"]].copy()
        vc["date"] = pd.to_datetime(vc["date"]).dt.date
        price_map = pd.Series(vc["market_price"].values, index=vc["date"].values).to_dict()

    # temp CSV
    tmp = tempfile.NamedTemporaryFile(prefix="icis_merged_", suffix=".csv", delete=False, mode="w", newline='', encoding="utf-8")
    tmp_path = tmp.name
    csv_writer = None
    sample_rows_list = []
    trade_mtm_records = []
    total_trades = len(trades_df)
    progress = st.progress(0)
    i = 0
    for rec in trades_df.to_dict("records"):
        i += 1
        try:
            daily_df = expand_trade_to_daily(rec, assume_side=assume_side, hub_country=hub_country)
        except Exception as e:
            logger.exception("Failed expanding trade %s: %s", rec.get("trade_id"), e)
            trade_mtm_records.append({"trade_id": rec.get("trade_id"), "trade_mtm": 0.0})
            progress.progress(int(i / total_trades * 100))
            continue
        if daily_df is None or daily_df.empty:
            trade_mtm_records.append({"trade_id": rec.get("trade_id"), "trade_mtm": 0.0})
            progress.progress(int(i / total_trades * 100))
            continue
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
        mtm_sum = 0.0
        if csv_writer is None:
            header = list(daily_df.columns) + ["market_price", "mtm_day"]
            csv_writer = csv.DictWriter(tmp, fieldnames=header)
            csv_writer.writeheader()
        for row in daily_df.to_dict("records"):
            d = row.get("date")
            mp = price_map.get(d, None)
            row["market_price"] = mp
            tp = row.get("trade_price")
            vol = row.get("daily_volume") or 0.0
            side = (row.get("trade_side") or assume_side).upper()
            mtm_day = None
            if mp is None or tp is None or (isinstance(mp, float) and math.isnan(mp)):
                mtm_day = ""
            else:
                try:
                    if side == "BUY":
                        mtm_day = (float(mp) - float(tp)) * float(vol)
                    else:
                        mtm_day = (float(tp) - float(mp)) * float(vol)
                except Exception:
                    mtm_day = ""
            row["mtm_day"] = mtm_day
            if isinstance(mtm_day, (int, float, np.floating)) and not (isinstance(mtm_day, float) and math.isnan(mtm_day)):
                try:
                    mtm_sum += float(mtm_day)
                except Exception:
                    pass
            csv_writer.writerow({k: row.get(k, "") for k in csv_writer.fieldnames})
            if len(sample_rows_list) < sample_rows:
                sample_rows_list.append(row)
        trade_mtm_records.append({"trade_id": rec.get("trade_id"), "trade_mtm": mtm_sum})
        progress.progress(int(i / total_trades * 100))
    tmp.close()
    merged_sample_df = pd.DataFrame(sample_rows_list)
    trade_mtm_df = pd.DataFrame(trade_mtm_records)
    portfolio_mtm = float(trade_mtm_df["trade_mtm"].sum()) if not trade_mtm_df.empty else 0.0
    return merged_sample_df, trade_mtm_df, portfolio_mtm, tmp_path

def process_trades_tradeonly_mtm(trades_df: pd.DataFrame, valuation_curve: pd.DataFrame, hub_country: str = "GB", assume_side: str = "BUY") -> Tuple[pd.DataFrame, float]:
    """
    Compute trade-level MTM only (no merged daily rows). Returns (trade_mtm_df, portfolio_mtm).
    Low-memory and faster.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(), 0.0
    # prepare date->price map
    price_map = {}
    if valuation_curve is not None and not valuation_curve.empty:
        vc = valuation_curve[["date","market_price"]].copy()
        vc["date"] = pd.to_datetime(vc["date"]).dt.date
        price_map = pd.Series(vc["market_price"].values, index=vc["date"].values).to_dict()

    trade_mtm_records = []
    total_trades = len(trades_df)
    progress = st.progress(0)
    i = 0
    for rec in trades_df.to_dict("records"):
        i += 1
        mtm_sum = 0.0
        try:
            daily_df = expand_trade_to_daily(rec, assume_side=assume_side, hub_country=hub_country)
        except Exception as e:
            logger.exception("Failed to expand trade %s: %s", rec.get("trade_id"), e)
            trade_mtm_records.append({"trade_id": rec.get("trade_id"), "trade_mtm": 0.0})
            progress.progress(int(i / total_trades * 100))
            continue
        if daily_df is None or daily_df.empty:
            trade_mtm_records.append({"trade_id": rec.get("trade_id"), "trade_mtm": 0.0})
            progress.progress(int(i / total_trades * 100))
            continue
        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
        for row in daily_df.to_dict("records"):
            d = row.get("date")
            mp = price_map.get(d, None)
            tp = row.get("trade_price")
            vol = row.get("daily_volume") or 0.0
            side = (row.get("trade_side") or assume_side).upper()
            if mp is None or tp is None or (isinstance(mp, float) and math.isnan(mp)):
                continue
            try:
                if side == "BUY":
                    mtm_sum += (float(mp) - float(tp)) * float(vol)
                else:
                    mtm_sum += (float(tp) - float(mp)) * float(vol)
            except Exception:
                continue
        trade_mtm_records.append({"trade_id": rec.get("trade_id"), "trade_mtm": mtm_sum})
        progress.progress(int(i / total_trades * 100))
    trade_mtm_df = pd.DataFrame(trade_mtm_records)
    portfolio_mtm = float(trade_mtm_df["trade_mtm"].sum()) if not trade_mtm_df.empty else 0.0
    return trade_mtm_df, portfolio_mtm

# ---------------------------
# MTM shock util
# ---------------------------
def apply_uniform_shock_and_compute_portfolio_mtm(merged: pd.DataFrame, shock_pct: float) -> float:
    if merged is None or merged.empty:
        return 0.0
    df = merged.copy()
    df["market_price_shock"] = df["market_price"] * (1.0 + shock_pct)
    mask_buy = df["trade_side"].fillna("BUY").str.upper() == "BUY"
    tp = pd.to_numeric(df["trade_price"], errors="coerce")
    vol = pd.to_numeric(df["daily_volume"], errors="coerce").fillna(0.0)
    mp_shock = df["market_price_shock"]
    mtm_shock = pd.Series(np.nan, index=df.index)
    valid = ~mp_shock.isna() & ~tp.isna()
    mtm_shock.loc[valid & mask_buy] = (mp_shock[valid & mask_buy] - tp[valid & mask_buy]) * vol[valid & mask_buy]
    mtm_shock.loc[valid & (~mask_buy)] = (tp[valid & (~mask_buy)] - mp_shock[valid & (~mask_buy)]) * vol[valid & (~mask_buy)]
    total = float(mtm_shock.sum(skipna=True)) if not mtm_shock.empty else 0.0
    return total

# ---------------------------
# Debug algebra helper
# ---------------------------
def q_to_missing_months_price_example(q_price: float, known_month_price: float, known_months_in_quarter: int = 1, missing_months: int = 2) -> float:
    total_needed = 3.0 * float(q_price)
    remaining = total_needed - float(known_month_price) * known_months_in_quarter
    per_missing = remaining / missing_months if missing_months > 0 else float('nan')
    return per_missing

# ---------------------------
# Streamlit UI wiring
# ---------------------------
st.set_page_config(page_title="ICIS Forward Gas Curves (Streaming + Trade-only MTM + Examples)", layout="wide")
st.title("ICIS Forward Gas Curves — Streaming Trades with Trade-only MTM & Examples")

# Sidebar auth & selectors
if "token" not in st.session_state:
    st.session_state["token"] = None
if "token_expires_at" not in st.session_state:
    st.session_state["token_expires_at"] = None

def token_is_valid() -> bool:
    t = st.session_state.get("token")
    if not t:
        return False
    exp = st.session_state.get("token_expires_at")
    if not exp:
        return True
    return datetime.utcnow() < (exp - timedelta(seconds=30))

if not token_is_valid():
    st.sidebar.markdown("Enter client_id & client_secret, or paste an access token.")
    client_id_val = st.sidebar.text_input("Client ID", value="", type="password")
    client_secret_val = st.sidebar.text_input("Client Secret", value="", type="password")
    pasted_token_val = st.sidebar.text_area("Paste access_token (optional)", value="")
    if st.sidebar.button("Authenticate (client_id/secret)"):
        if not client_id_val or not client_secret_val:
            st.sidebar.error("Provide both fields")
        else:
            try:
                tok = get_token_and_expiry(client_id_val, client_secret_val)
                st.session_state["token"] = tok["access_token"]
                st.session_state["token_expires_at"] = tok["expires_at"]
                st.sidebar.success("Authenticated.")
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"Auth failed: {e}")
    if st.sidebar.button("Use pasted token"):
        if not pasted_token_val:
            st.sidebar.error("Paste token first")
        else:
            st.session_state["token"] = pasted_token_val
            st.session_state["token_expires_at"] = None
            st.sidebar.success("Token stored.")
            st.experimental_rerun()
else:
    st.sidebar.success("Token available")
    if st.sidebar.button("Logout (clear token)"):
        st.session_state["token"] = None
        st.session_state["token_expires_at"] = None
        st.experimental_rerun()

token = st.session_state.get("token")
if not token:
    st.info("Authenticate to continue.")
    st.stop()

# Load specs
with st.spinner("Loading specifications..."):
    try:
        price_specs = fetch_spec_price_assessments(token)
        trade_specs = fetch_trade_specifications(token)
    except Exception as e:
        st.error(f"Failed to load specifications: {e}")
        st.stop()

price_spec_map = {s.get("id"): s.get("attributes", {}).get("name", f"{s.get('id')}") for s in price_specs}
trade_spec_map = {s.get("id"): s.get("attributes", {}).get("name", f"{s.get('id')}") for s in trade_specs}
trade_display_by_id = {tid: f"{trade_spec_map[tid]} | id={tid}" for tid in trade_spec_map.keys()}
trade_display_list = [trade_display_by_id[tid] for tid in trade_display_by_id.keys()]
display_to_id = {trade_display_by_id[tid]: tid for tid in trade_display_by_id.keys()}

# Sidebar options
st.sidebar.header("Selectors & Options")
selected_price_spec_name = st.sidebar.selectbox("PriceAssessment specification (for curve & history)", options=[""] + list(price_spec_map.values()))
selected_trade_spec_displays = st.sidebar.multiselect("Trade specification(s) (for trades)", options=trade_display_list)

st.sidebar.markdown("createdForDate range (YYYYMMDD used for API)")
start_date = st.sidebar.date_input("createdForDateStart", value=datetime.utcnow().date() - timedelta(days=90))
end_date = st.sidebar.date_input("createdForDateEnd", value=datetime.utcnow().date())
start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

st.sidebar.header("Curve & processing")
curve_source = st.sidebar.selectbox("Curve source for valuation", options=["Allocation (priority)","Assessments period-average","Trades-derived (period VWAP)","Overlay (show all)"], index=0)
assess_method = st.sidebar.selectbox("Assessments aggregation", options=["mean","median","latest"], index=0)
trades_method = st.sidebar.selectbox("Trades aggregation", options=["VWAP","average","median","trimmed"], index=0)
trim_pct = st.sidebar.slider("Trim % (for trimmed mean)", min_value=0.0, max_value=0.25, value=0.1, step=0.01)
enforce_arbitrage = st.sidebar.checkbox("Enforce arbitrage-free monthly derivation", value=True)

st.sidebar.markdown("WDNW & shaping")
wdnw_mode = st.sidebar.selectbox("WDNW weighting", options=["Equal","DA-weighted"], index=1)
da_weight = 0.5
if wdnw_mode == "DA-weighted":
    da_weight = st.sidebar.slider("DA weight (fraction)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

apply_icis_shaping = st.sidebar.checkbox("Apply strict ICIS shaping (5-year)", value=False)
country_code = st.sidebar.text_input("Holiday country code (e.g., GB)", value="GB")
fetch_history_for_shaping = st.sidebar.checkbox("Fetch 5-year history for shaping (may be slow)", value=False)
front_month_mode = st.sidebar.selectbox("Front-month mode (BOM normalization)", options=["calendar","front_contract"], index=1)

st.sidebar.header("Trades & MTM mode")
mtm_mode = st.sidebar.selectbox("MTM computation mode", options=["streaming_merged","trade_only"], index=0)
st.sidebar.markdown(" - streaming_merged: writes merged daily rows to disk and provides downloadable CSV (slower, full detail).")
st.sidebar.markdown(" - trade_only: computes only trade-level MTM summary (fast, low memory).")

missing_side_default = st.sidebar.selectbox("Default tradeSide for missing values", options=["BUY","SELL"], index=0)
curve_choice_for_mtm = st.sidebar.selectbox("Use curve for valuation (MTM)", options=["Selected curve","Allocation (priority)"], index=0)

# Manual actions: build curves & optionally fetch 5-year history
if st.sidebar.button("Build PriceAssessment curve"):
    if not selected_price_spec_name:
        st.sidebar.error("Select PriceAssessment specification first.")
    else:
        selected_price_spec_id = {v:k for k,v in price_spec_map.items()}[selected_price_spec_name]
        with st.spinner("Fetching priceAssessments and building daily curve(s)..."):
            try:
                combined_pa = fetch_price_assessments(token, selected_price_spec_id, start_str, end_str)
            except Exception as e:
                st.error(f"Failed to fetch priceAssessments: {e}")
                st.stop()
        # Parse the assessments and show raw table (requested change)
        df_assess = parse_assessments_with_periods(combined_pa)
        st.session_state["df_assess"] = df_assess
        st.subheader("PriceAssessments (parsed raw rows)")
        # show a reasonably sized sample but allow user to scroll
        st.dataframe(df_assess.head(1000))
        # provide download with unique key
        st.download_button("Download parsed PriceAssessments CSV", df_assess.to_csv(index=False).encode("utf-8"), file_name="assessments_parsed.csv", mime="text/csv", key="download_assessments_parsed")

        st.success("Assessments fetched.")
        alloc_daily = build_daily_curve_allocation(df_assess)
        st.session_state["alloc_curve"] = alloc_daily
        period_prices_assess = compute_period_prices_from_assessments(df_assess, method=assess_method)
        if enforce_arbitrage:
            af_months = enforce_arbitrage_free_monthly(period_prices_assess)
            pr_rows = []
            for _, r in af_months.iterrows():
                y = int(r["year"]); m = int(r["month"])
                start = pd.Timestamp(year=y, month=m, day=1)
                end = (start + pd.DateOffset(months=1) - pd.Timedelta(days=1))
                pr_rows.append({
                    "period_id": f"{y}{m:02d}",
                    "period_price": r["period_price"],
                    "period_startDate": start.isoformat(),
                    "period_endDate": end.isoformat(),
                    "period_label": f"{start.strftime('%B %Y')}"
                })
            af_period_prices_df = pd.DataFrame(pr_rows)
            assess_daily = expand_period_prices_to_daily(af_period_prices_df)
        else:
            assess_daily = expand_period_prices_to_daily(period_prices_assess)
        if "period_price" in assess_daily.columns:
            assess_daily = assess_daily.rename(columns={"period_price":"market_price"})
        else:
            assess_daily["market_price"] = np.nan
        st.session_state["assess_daily"] = assess_daily
        # Build a period map for later joining (period_id -> relativePeriodCode/label from assessments)
        try:
            period_map = {}
            if not df_assess.empty and "period_id" in df_assess.columns:
                # prefer relativePeriodCode, fallback to default label
                for _, r in df_assess.iterrows():
                    pid = r.get("period_id")
                    code = r.get("period_relativePeriodCode") or r.get("period_defaultPeriodLabel")
                    if pid is not None:
                        period_map[pid] = code
            st.session_state["period_map"] = period_map
        except Exception:
            st.session_state["period_map"] = {}
        st.success("Curves built and stored in session.")
        if fetch_history_for_shaping:
            hist_start = (datetime.utcnow().date() - timedelta(days=365*5))
            with st.spinner("Fetching 5-year historical assessments for shaping..."):
                try:
                    combined_hist = fetch_price_assessments(token, selected_price_spec_id, hist_start.strftime("%Y%m%d"), end_str)
                    df_hist_assess = parse_assessments_with_periods(combined_hist)
                    st.session_state["df_hist_assess"] = df_hist_assess
                    st.session_state["hist_alloc"] = build_daily_curve_allocation(df_hist_assess)
                    st.session_state["period_prices_hist"] = compute_period_prices_from_assessments(df_hist_assess)
                    st.success("Historical assessments fetched for shaping.")
                except Exception as e:
                    st.warning(f"Failed to fetch full 5-year history: {e}")

# require baseline curves
if "alloc_curve" not in st.session_state and "assess_daily" not in st.session_state:
    st.info("Build a PriceAssessment curve first (sidebar).")
    st.stop()

# Combined preview
st.subheader("Curve previews & combined table")
alloc = st.session_state.get("alloc_curve")
assess = st.session_state.get("assess_daily")
trades_derived = st.session_state.get("trades_derived_daily")
if isinstance(trades_derived, pd.DataFrame) and not trades_derived.empty and "period_price" in trades_derived.columns and "market_price" not in trades_derived.columns:
    trades_derived = trades_derived.rename(columns={"period_price":"market_price"})
    st.session_state["trades_derived_daily"] = trades_derived

def norm(df, col):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["date", col])
    out = df[["date","market_price"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.rename(columns={"market_price":col})
    return out

alloc_norm = norm(alloc, "allocation_price")
assess_norm = norm(assess, "assess_price")
trades_norm = norm(st.session_state.get("trades_derived_daily"), "trades_price")

date_collections = []
for ddf in (alloc_norm, assess_norm, trades_norm):
    if isinstance(ddf, pd.DataFrame) and not ddf.empty:
        date_collections.append(pd.to_datetime(ddf["date"]).dt.date)
if date_collections:
    concat_dates = pd.concat(date_collections)
    min_date = concat_dates.min()
    max_date = concat_dates.max()
    all_dates = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D").date})
else:
    all_dates = pd.DataFrame({"date": []})

combined = all_dates.merge(alloc_norm, on="date", how="left").merge(assess_norm, on="date", how="left").merge(trades_norm, on="date", how="left")
st.dataframe(combined.head(200))
st.download_button("Download combined curve CSV", combined.to_csv(index=False).encode("utf-8"), file_name="combined_curves.csv", mime="text/csv", key="download_combined_curves")

# Chart overlay
st.subheader("Curve overlay")
fig = go.Figure()
if not alloc_norm.empty:
    fig.add_trace(go.Scatter(x=pd.to_datetime(alloc_norm["date"]), y=alloc_norm["allocation_price"], name="Allocation (priority)", line=dict(color="gray")))
if not assess_norm.empty:
    fig.add_trace(go.Scatter(x=pd.to_datetime(assess_norm["date"]), y=assess_norm["assess_price"], name="Assessments", line=dict(color="blue")))
if not trades_norm.empty:
    fig.add_trace(go.Scatter(x=pd.to_datetime(trades_norm["date"]), y=trades_norm["trades_price"], name="Trades-derived", line=dict(color="orange")))
fig.update_layout(title="Curve comparison", xaxis_title="Date", yaxis_title="Price", height=420)
st.plotly_chart(fig, use_container_width=True)

# Trades fetch & MTM (streaming or trade-only)
st.header("Trades: fetch & compute MTM")
if st.button("Fetch trades and compute MTM"):
    refresh_time = datetime.utcnow()
    if not selected_trade_spec_displays:
        st.error("Select at least one Trade Specification in the sidebar.")
        st.stop()
    spec_ids = [display_to_id[disp] for disp in selected_trade_spec_displays if disp in display_to_id]
    if not spec_ids:
        st.error("Failed to determine tradeSpecification ids from selection.")
        st.stop()
    with st.spinner("Fetching trades..."):
        try:
            combined_trades = fetch_trades(token, spec_ids, start_str, end_str)
        except Exception as e:
            st.error(f"Failed to fetch trades: {e}")
            st.stop()
    trades_df = parse_trades_with_periods(combined_trades)
    st.session_state["trades_raw"] = trades_df
    st.subheader("Trades (raw sample)")
    st.dataframe(trades_df.head(200))
    if trades_df.empty:
        st.warning("No trades returned for the selected specification id(s) and date range.")
        st.stop()

    # Precompute trades-derived daily curve for completeness
    period_prices_trades = compute_period_prices_from_trades(trades_df, method=trades_method if trades_method != "trimmed" else "trimmed", trim_pct=trim_pct)
    trades_derived_daily = expand_period_prices_to_daily(period_prices_trades)
    if "period_price" in trades_derived_daily.columns:
        trades_derived_daily = trades_derived_daily.rename(columns={"period_price":"market_price"})
    else:
        trades_derived_daily["market_price"] = np.nan
    if "date" in trades_derived_daily.columns:
        trades_derived_daily["date"] = pd.to_datetime(trades_derived_daily["date"]).dt.date
    st.session_state["trades_derived_daily"] = trades_derived_daily

    # choose valuation curve
    if curve_source == "Allocation (priority)":
        valuation_curve = choose_df(st.session_state.get("alloc_curve"), pd.DataFrame())
    elif curve_source == "Assessments period-average":
        valuation_curve = choose_df(st.session_state.get("assess_daily"), pd.DataFrame())
    elif curve_source == "Trades-derived (period VWAP)":
        valuation_curve = choose_df(st.session_state.get("trades_derived_daily"), pd.DataFrame())
    else:
        valuation_curve = choose_df(st.session_state.get("assess_daily"), st.session_state.get("alloc_curve"))

    # apply shaping if requested
    if apply_icis_shaping:
        hist_alloc = st.session_state.get("hist_alloc")
        if hist_alloc is None or hist_alloc.empty:
            st.warning("5-year history required for strict ICIS shaping — enable 'Fetch 5-year history' and rebuild.")
        else:
            hist_daily = hist_alloc[["date","market_price"]].rename(columns={"market_price":"price"})
            reference_month = datetime.utcnow().month
            period_prices_hist = st.session_state.get("period_prices_hist")
            bom_dow_factors = compute_bom_weekday_factors_from_history(hist_daily, reference_month, window_years=5, front_month_mode=front_month_mode, period_prices_hist=period_prices_hist)
            st.write("BOM weekday shaping factors (Mon..Sun):", [round(bom_dow_factors[i]*100,4) for i in range(7)])
            period_prices_hist_local = period_prices_hist if period_prices_hist is not None else compute_period_prices_from_assessments(st.session_state.get("df_hist_assess", pd.DataFrame()))
            month_of_year_factors = compute_monthly_granularity_shaping_from_history(period_prices_hist_local, window_years=5)
            st.write("Month-of-year factors:", {m: round(month_of_year_factors.get(m,1.0),4) for m in range(1,13)})
            if not valuation_curve.empty:
                valuation_curve_shaped = valuation_curve.copy()
                valuation_curve_shaped["dow"] = pd.to_datetime(valuation_curve_shaped["date"]).dt.weekday
                valuation_curve_shaped["factor"] = valuation_curve_shaped["dow"].map(lambda x: bom_dow_factors.get(int(x), 1.0))
                valuation_curve_shaped["market_price"] = valuation_curve_shaped["market_price"] * valuation_curve_shaped["factor"]
                orig_mean = valuation_curve["market_price"].mean(skipna=True)
                new_mean = valuation_curve_shaped["market_price"].mean(skipna=True)
                if new_mean and not math.isnan(new_mean) and new_mean != 0:
                    valuation_curve_shaped["market_price"] = valuation_curve_shaped["market_price"] * (orig_mean / new_mean)
                valuation_curve = valuation_curve_shaped.drop(columns=["dow","factor"])
                st.session_state["valuation_curve_shaped"] = valuation_curve
                st.success("Applied shaping to valuation curve.")

    # Expand trades -> daily rows (we will always expand for insights even if trade_only mode)
    daily_rows = []
    for _, tr in trades_df.iterrows():
        daily_rows.append(expand_trade_to_daily(tr.to_dict(), assume_side=missing_side_default, hub_country=country_code))
    trades_daily_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame(columns=["trade_id","date","daily_volume","trade_price","trade_side"])
    st.subheader("Expanded trades (daily) sample")
    st.dataframe(trades_daily_df.head(200))

    # choose mode and compute MTM
    merged_sample_df = pd.DataFrame()
    trade_mtm_df = pd.DataFrame()
    merged_csv_path = ""
    portfolio_mtm = 0.0
    if mtm_mode == "streaming_merged":
        merged_sample_df, trade_mtm_df, portfolio_mtm, merged_csv_path = process_trades_streaming_and_compute_mtm(
            trades_df, valuation_curve, hub_country=country_code, assume_side=missing_side_default, sample_rows=200
        )
        st.subheader("Per-trade MTM (sample)")
        # We will merge period info below and display labeled table
        st.write(f"Portfolio MTM: {portfolio_mtm:.2f}")
        st.subheader("Merged daily sample (first rows)")
        if merged_sample_df is None or merged_sample_df.empty:
            st.write("No merged sample available.")
        else:
            st.dataframe(merged_sample_df.head(200))
        # use unique key for download (filename included)
        if merged_csv_path and os.path.exists(merged_csv_path):
            with open(merged_csv_path, "rb") as f:
                data = f.read()
            st.download_button("Download full merged daily MTM CSV", data, file_name=os.path.basename(merged_csv_path), mime="text/csv", key=f"download_merged_{os.path.basename(merged_csv_path)}")
        else:
            st.warning("Merged CSV not available.")
    else:
        # trade_only
        trade_mtm_df, portfolio_mtm = process_trades_tradeonly_mtm(trades_df, valuation_curve, hub_country=country_code, assume_side=missing_side_default)
        st.subheader("Trade-level MTM (summary)")
        st.write(f"Portfolio MTM: {portfolio_mtm:.2f}")
        # allow download of trade summary with unique key
        st.download_button("Download trade-level MTM CSV", trade_mtm_df.to_csv(index=False).encode("utf-8"), file_name="trade_level_mtm.csv", mime="text/csv", key="download_trade_level_mtm")

    # --- JOIN PERIOD INFO FROM TRADE 'included' OR FROM PRICEASSESSMENTS (if available) ---
    # trades_df contains period fields from its included objects (period_id, period_relativePeriodCode, etc).
    # The price assessments parsed earlier (st.session_state["df_assess"]) contain canonical period info that can be used
    # to map period_id -> relativePeriodCode/label. We'll prefer the priceAssessments mapping if available.
    try:
        period_map = st.session_state.get("period_map", {}) if "period_map" in st.session_state else {}
        # Build a small trades-period dataframe for merging
        if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
            trades_periods = trades_df[["trade_id", "period_id", "period_relativePeriodCode", "period_relativePeriodLabel"]].drop_duplicates(subset=["trade_id"])
        else:
            trades_periods = pd.DataFrame(columns=["trade_id", "period_id", "period_relativePeriodCode", "period_relativePeriodLabel"])
        if not trade_mtm_df.empty:
            # merge trade-level MTM with trade->period mapping
            trade_mtm_labeled = trade_mtm_df.merge(trades_periods, on="trade_id", how="left")
            # derive final period label: prefer period_map (from PriceAssessments), then trade's relativePeriodCode, then period_relativePeriodLabel, else trade_id
            def _final_label(row):
                pid = row.get("period_id")
                if pid and pid in period_map and period_map.get(pid):
                    return period_map.get(pid)
                # fallback to trade's own included period code/label
                prc = row.get("period_relativePeriodCode") or row.get("period_relativePeriodLabel")
                if prc:
                    return prc
                return str(row.get("trade_id"))
            trade_mtm_labeled["period_label"] = trade_mtm_labeled.apply(_final_label, axis=1)
            # store for display and downloads
            st.subheader("Trade MTM labeled by Period (sample)")
            # show columns: period_label, period_id, trade_mtm, trade_id (for reference)
            display_cols = ["period_label", "period_id", "trade_id", "trade_mtm"]
            st.dataframe(trade_mtm_labeled[display_cols].head(200))
            # use 'period_label' in later plots (e.g., scatter hover text)
        else:
            trade_mtm_labeled = pd.DataFrame()
    except Exception:
        trade_mtm_labeled = pd.DataFrame()

    # --- DYNAMIC INSIGHTS & EXPLANATION (added) ---
    # Build a merged_for_insights (full expanded trades merged with valuation curve) so we can compute diagnostics and charts.
    try:
        valuation_curve_for_merge = valuation_curve.copy() if valuation_curve is not None else pd.DataFrame(columns=["date","market_price"])
        valuation_curve_for_merge["date"] = pd.to_datetime(valuation_curve_for_merge["date"]).dt.date
        merged_for_insights = trades_daily_df.copy()
        merged_for_insights["date"] = pd.to_datetime(merged_for_insights["date"]).dt.date
        merged_for_insights = merged_for_insights.merge(valuation_curve_for_merge[["date","market_price"]], on="date", how="left")
        # compute mtm_day same way as processors
        def _mtm_row_local(r):
            mp = r.get("market_price")
            tp = r.get("trade_price")
            vol = r.get("daily_volume") or 0.0
            side = (r.get("trade_side") or missing_side_default).upper()
            if pd.isna(mp) or tp is None:
                return np.nan
            try:
                if side == "BUY":
                    return (float(mp) - float(tp)) * float(vol)
                else:
                    return (float(tp) - float(mp)) * float(vol)
            except Exception:
                return np.nan
        merged_for_insights["mtm_day"] = merged_for_insights.apply(_mtm_row_local, axis=1)
    except Exception:
        merged_for_insights = pd.DataFrame()

    # Prepare insights structure
    try:
        insights = compute_insights(merged_for_insights, trade_mtm_df if 'trade_mtm_df' in locals() else pd.DataFrame(), trades_daily_df)
    except Exception as e:
        logger.exception("Failed to compute insights: %s", e)
        insights = {}

    st.markdown("## Dynamic MTM Insights (auto-updated)")
    cols = st.columns(3)
    # show metrics; guard missing keys
    portfolio_val = insights.get("portfolio_mtm", float(portfolio_mtm) if 'portfolio_mtm' in locals() else 0.0)
    num_trades = insights.get("num_trades", int(trade_mtm_df.shape[0]) if 'trade_mtm_df' in locals() else 0)
    pos = insights.get("positive_trades", 0)
    neg = insights.get("negative_trades", 0)
    cols[0].metric("Portfolio MTM", f"{portfolio_val:.2f}")
    cols[1].metric("Trades evaluated", f"{num_trades}")
    cols[2].metric("Trades (pos / neg)", f"{pos} / {neg}")

    # Dynamic explanatory text
    st.markdown("### What these MTM numbers mean (dynamic explanation)")
    expl_lines = []
    expl_lines.append(f"- MTM refresh time (UTC): **{refresh_time.strftime('%Y-%m-%d %H:%M:%S')}**")
    expl_lines.append(f"- Valuation curve source: **{curve_source}**")
    if curve_choice_for_mtm == "Allocation (priority)":
        expl_lines.append("- Curve chosen for MTM: Allocation (priority) curve (most-specific assessment priority).")
    else:
        expl_lines.append(f"- Curve chosen for MTM: selected computed curve (after shaping if applied).")
    expl_lines.append(f"- Portfolio MTM is the arithmetic sum of per-trade MTMs computed as daily contributions where MarketPrice is available.")
    expl_lines.append(f"- Number of trades evaluated: **{num_trades}**. Trades/days with missing market prices are ignored for those days.")
    expl_lines.append(f"- Daily volume allocation: if `volume` is present it is used as per-day; otherwise `totalVolume` divided by the number of days in the period.")
    expl_lines.append(f"- Missing tradeSide defaults to **{missing_side_default}**.")
    missing_market_rows = insights.get("missing_market_rows", 0)
    expl_lines.append(f"- Missing market-price rows: **{missing_market_rows}** (those rows do not contribute to MTM).")
    expl_lines.append("- Shock scenario applies a uniform percentage change to all market prices (see scenario analyser below).")
    expl_text_block = "\n".join(expl_lines)
    st.info(expl_text_block)

    # Charts: 1) cumulative daily MTM; 2) histogram of trade-level MTM; 3) pie sign breakdown; 4) exposure by month; 5) breakeven scatter
    st.markdown("## MTM Visuals")

    # 1) cumulative daily MTM time-series
    try:
        if not merged_for_insights.empty:
            daily = merged_for_insights.groupby(pd.to_datetime(merged_for_insights["date"]).dt.date)["mtm_day"].sum().sort_index()
            if not daily.empty:
                cumulative = daily.cumsum()
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Bar(x=list(daily.index), y=daily.values, name="Daily MTM", marker_color="lightsalmon"))
                fig_cum.add_trace(go.Scatter(x=list(cumulative.index), y=cumulative.values, mode="lines+markers", name="Cumulative MTM", line=dict(color="firebrick", width=3)))
                fig_cum.update_layout(title="Daily MTM and Cumulative MTM", xaxis_title="Date", yaxis_title="MTM", height=440)
                st.plotly_chart(fig_cum, use_container_width=True)
            else:
                st.write("No daily MTM points to show time-series.")
        else:
            st.write("No merged data to plot daily MTM (possible in trade_only with no valuation coverage).")
    except Exception as e:
        logger.exception("Cumulative MTM chart failed: %s", e)

    # 2) Histogram of per-trade MTM
    try:
        if 'trade_mtm_df' in locals() and not trade_mtm_df.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=trade_mtm_df["trade_mtm"].values, nbinsx=40, marker_color="steelblue"))
            fig_hist.update_layout(title="Distribution of Trade-level MTM", xaxis_title="Trade MTM", yaxis_title="Count", height=340)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.write("No trade-level MTM to show histogram.")
    except Exception as e:
        logger.exception("Trade MTM histogram failed: %s", e)

    # 3) Pie chart of positive vs negative trade counts
    try:
        pos = insights.get("positive_trades", 0)
        neg = insights.get("negative_trades", 0)
        neu = max(0, insights.get("num_trades", 0) - pos - neg)
        if (pos + neg + neu) > 0:
            fig_pie = go.Figure(go.Pie(labels=["Positive MTM trades", "Negative MTM trades", "Zero/Unknown"], values=[pos, neg, neu], hole=0.4))
            fig_pie.update_layout(title="Trade MTM sign breakdown", height=340)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("No trade sign breakdown to show.")
    except Exception as e:
        logger.exception("Pie chart failed: %s", e)

    # 4) Exposure by month - bar chart
    try:
        exp = insights.get("exposure_by_month", {})
        if exp:
            exp_df = pd.DataFrame(list(exp.items()), columns=["month", "exposure"]).sort_values("month")
            fig_exp = go.Figure(go.Bar(x=exp_df["month"], y=exp_df["exposure"], marker_color="mediumpurple"))
            fig_exp.update_layout(title="Exposure by Month (market_price * daily_volume aggregated)", xaxis_title="Month", yaxis_title="Exposure", height=360)
            st.plotly_chart(fig_exp, use_container_width=True)
        else:
            st.write("No exposure-by-month data available.")
    except Exception as e:
        logger.exception("Exposure chart failed: %s", e)

    # 5) Scatter: breakeven market price vs trade MTM (use period_label where available)
    try:
        tb_df = insights.get("trade_breakeven", pd.DataFrame())
        if not tb_df.empty and 'trade_mtm_labeled' in locals() and not trade_mtm_labeled.empty:
            tb = trade_mtm_labeled.merge(tb_df, how="left", on="trade_id")
            tb_nonnull = tb.dropna(subset=["breakeven_market_price"])
            if not tb_nonnull.empty:
                # use period_label as hover text (we created that earlier)
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(x=tb_nonnull["breakeven_market_price"], y=tb_nonnull["trade_mtm"], mode="markers", text=tb_nonnull["period_label"].astype(str), marker=dict(size=8, color="teal")))
                fig_scatter.update_layout(title="Trade breakeven market price vs trade MTM (period label shown)", xaxis_title="Breakeven market price", yaxis_title="Trade MTM", height=420)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.write("No breakeven data available to plot scatter.")
        else:
            st.write("Breakeven data not available for scatter.")
    except Exception as e:
        logger.exception("Breakeven scatter failed: %s", e)

    # Shock / scenario analyser (reuse existing util)
    st.markdown("### Scenario / Shock analyser (uniform shock to market curve)")
    shock_pct = st.slider("Uniform market shock (%) (positive = price up)", min_value=-50, max_value=+50, value=5, step=1)
    shock_decimal = shock_pct / 100.0
    shocked_portfolio_mtm = apply_uniform_shock_and_compute_portfolio_mtm(merged_for_insights, shock_decimal) if not merged_for_insights.empty else 0.0
    st.metric("Portfolio MTM under shock", f"{shocked_portfolio_mtm:.2f}", delta=f"{shocked_portfolio_mtm - portfolio_val:.2f}")

    # small audit info and link to downloads (explain where to find CSVs)
    st.markdown("### Audit & next steps")
    st.markdown(
        "- Use the download buttons above to export the *expanded daily rows* and *trade-level MTM* for audit.\n"
        "- Check flagged missing dates if coverage gaps materially affect MTM.\n"
        "- To reproduce, note the selected PriceAssessment spec, trade spec ids, createdForDate range, shaping params and shock applied."
    )

    # Downloads (restate available downloads) - give unique keys for each
    st.subheader("Downloads")
    if "df_assess" in st.session_state:
        st.download_button("Download priceAssessments (assessments) CSV", st.session_state["df_assess"].to_csv(index=False).encode("utf-8"), file_name="assessments.csv", mime="text/csv", key="download_assessments_2")
    if "alloc_curve" in st.session_state:
        st.download_button("Download allocation curve CSV", st.session_state["alloc_curve"].to_csv(index=False).encode("utf-8"), file_name="allocation_curve.csv", key="download_alloc_curve", mime="text/csv")
    if "assess_daily" in st.session_state:
        st.download_button("Download daily curve CSV", st.session_state["assess_daily"].to_csv(index=False).encode("utf-8"), file_name="daily_curve.csv", key="download_daily_curve", mime="text/csv")
    if "trades_raw" in st.session_state:
        st.download_button("Download trades raw CSV", st.session_state["trades_raw"].to_csv(index=False).encode("utf-8"), file_name="trades_raw.csv", key="download_trades_raw", mime="text/csv")
    if 'trades_daily_df' in locals() and not trades_daily_df.empty:
        st.download_button("Download trades_daily CSV", trades_daily_df.to_csv(index=False).encode("utf-8"), file_name="trades_daily.csv", key="download_trades_daily", mime="text/csv")
    if mtm_mode == "streaming_merged" and merged_csv_path and os.path.exists(merged_csv_path):
        with open(merged_csv_path, "rb") as f:
            data = f.read()
        st.download_button("Download full merged daily MTM CSV", data, file_name=os.path.basename(merged_csv_path), mime="text/csv", key=f"download_merged_final_{os.path.basename(merged_csv_path)}")
    if 'trade_mtm_df' in locals() and not trade_mtm_df.empty:
        st.download_button("Download trade-level MTM CSV", trade_mtm_df.to_csv(index=False).encode("utf-8"), file_name="trade_level_mtm.csv", mime="text/csv", key="download_trade_level_mtm_2")

# ---------------------------
# Debug / Test panel (examples)
# ---------------------------
st.markdown("---")
st.subheader("Debug / Test panel (audit examples)")

with st.expander("Q→M example (August 2025 calculation)"):
    st.markdown("ICIS algebra example: ((3 * Q3) - July) / 2")
    q_price = st.number_input("Quarter Q3 2025 price (€/MWh)", value=48.563, format="%.6f", key="q_price")
    m06_price = st.number_input("Known July 2025 (M06) price (€/MWh)", value=48.488, format="%.6f", key="m06_price")
    if st.button("Compute August 2025 price", key="compute_aug"):
        aug_price = q_to_missing_months_price_example(q_price, m06_price, known_months_in_quarter=1, missing_months=2)
        st.write(f"Computed August 2025 price = ((3 * {q_price}) - {m06_price}) / 2 = {aug_price:.6f} €/MWh")

with st.expander("WDNW & DA example"):
    rep_date = st.date_input("Report date for WDNW example", value=datetime.utcnow().date(), key="rep_date")
    hub_cc = st.text_input("Hub country code (for working days)", value="GB", key="hub_cc")
    if st.button("Show WDNW days and DA (example)", key="wdnw_example"):
        wdnw_days = next_week_working_days(rep_date, hub_cc)
        da_day = next_working_gas_day(rep_date, hub_cc)
        st.write(f"Report date: {rep_date}, Day‑Ahead (DA): {da_day}")
        st.write("WDNW days:", wdnw_days)
        alloc = st.session_state.get("alloc_curve")
        assess = st.session_state.get("assess_daily")
        tradesd = st.session_state.get("trades_derived_daily")
        days_eq, price_eq = derive_wdnw_days_and_price(rep_date, alloc, assess, tradesd, hub_cc, wdnw_mode="Equal")
        days_da, price_da = derive_wdnw_days_and_price(rep_date, alloc, assess, tradesd, hub_cc, wdnw_mode="DA-weighted", da_weight=0.5)
        st.write(f"WDNW equal-weight price: {price_eq}")
        st.write(f"WDNW DA-weighted price (DA weight=0.5): {price_da}")

with st.expander("January shaping factors (BOM weekday factors)"):
    if st.button("Compute January factors (requires 5-year history)", key="jan_shaping"):
        hist_alloc = st.session_state.get("hist_alloc")
        if hist_alloc is None or hist_alloc.empty:
            st.warning("5-year allocation history not available. Enable 'Fetch 5-year history' and rebuild PriceAssessment curve.")
        else:
            hist_daily = hist_alloc[["date","market_price"]].rename(columns={"market_price":"price"})
            period_prices_hist_local = st.session_state.get("period_prices_hist")
            factors = compute_bom_weekday_factors_from_history(hist_daily, reference_month=1, window_years=5, front_month_mode=front_month_mode, period_prices_hist=period_prices_hist_local)
            st.write("Shaping factors for January (Mon..Sun) in % (normalized to mean=100%):")
            st.write({["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][i]: f"{factors[i]*100:.4f}%" for i in range(7)})

st.markdown("---")
st.markdown(
    "Notes:\n"
    "- 'trade_only' mode computes trade-level MTM only (fast, low memory). Use it for very large trade volumes.\n"
    "- 'streaming_merged' mode writes merged daily rows to a temporary CSV to provide full daily detail (slower, but full detail).\n"
    "- Strict ICIS shaping requires fetching 5-year history (enable 'Fetch 5-year history' when building the PriceAssessment curve).\n"
    "- For accurate holiday rules install python-holidays: pip install holidays\n"
)