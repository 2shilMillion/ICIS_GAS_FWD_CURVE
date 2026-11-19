#!/usr/bin/env python3
"""
ICIS Forward Gas Curves — Full Streamlit app with Implied Volatility Analysis
and Location Spread features.

This version includes:
- PriceAssessment parsing and per-period handling
- Priority-based allocation and preserve-average assignment
- Trades parsing, explicit "Fetch trades" button, expansion to daily,
  and period aggregation (VWAP/median/trimmed)
- MTM computation with expanded trades vs chosen valuation curve
- MTM insights: daily & cumulative PnL, distribution, rolling volatility, VaR
- Top contributors aggregated by period and stacked PnL by period
- Implied Volatility Analysis:
  - Realized volatility by period
  - Volume-weighted volatility
  - Bid-ask spread analysis
  - Moneyness and IV proxy
  - Rolling curve volatility
  - Term structure volatility
- Location spread analysis:
  - compare multiple hubs
  - only uses overlapping delivery periods
  - builds a forward-looking spread curve by matched periods
  - explains what the spread means

Run:
    pip install streamlit requests pandas numpy plotly holidays
    streamlit run API_Gas_ICIS_fwd_curve_IV_2.py
"""
from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import logging
import zoneinfo
import os
import tempfile
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

# ---- Configuration ----
BASE_API = "https://api.icis.com/energy/v1"
SPEC_PRICEASSESSMENTS_ENDPOINT = urljoin(BASE_API + "/", "priceAssessments/specifications")
PRICE_ASSESSMENTS_ENDPOINT = urljoin(BASE_API + "/", "priceAssessments")
SPEC_TRADES_ENDPOINT = urljoin(BASE_API + "/", "trades/specifications")
TRADES_ENDPOINT = urljoin(BASE_API + "/", "trades")
TOKEN_URL = "https://api.icis.com/connect/token"
PAGE_SIZE = 1000

PRIORITY_ORDER = {
    "DA": 1, "DAY-AHEAD": 1, "DAY AHEAD": 1,
    "WEEKEND": 2, "WKND": 2,
    "WDNW": 3, "WD/NW": 3,
    "BOM": 4, "BALANCE": 4, "BALANCE-OF-MONTH": 4,
    "MONTH": 5, "MONTHLY": 5, "M": 5,
    "Q": 6, "QUARTER": 6,
    "SEASON": 7, "WINTER": 7, "SUMMER": 7,
    "YEAR": 8, "GAS YEAR": 8,
}

GAS_DAY_START_OVERRIDES = {"GB": 5, "UA": 5}
DEFAULT_GAS_DAY_START = 6

HUB_TIMEZONES = {
    "GB": "Europe/London",
    "NL": "Europe/Amsterdam",
    "DE": "Europe/Berlin",
    "FR": "Europe/Paris",
    "UA": "Europe/Kyiv",
}

# ---------------------------
# Helpers
# ---------------------------
def rerun_app() -> None:
    if hasattr(st, "rerun") and callable(st.rerun):
        st.rerun()
    elif hasattr(st, "experimental_rerun") and callable(st.experimental_rerun):
        st.experimental_rerun()

def _safe_unlink(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        logger.debug("Failed to delete temp file %s", path)

def st_download_df(label: str, df: pd.DataFrame, file_name: str, mime: str = "text/csv", encoding: str = "utf-8", tmp_life_seconds: int = 60):
    if df is None:
        st.warning("Nothing to download (DataFrame is None).")
        return

    if isinstance(df, pd.DataFrame) and df.empty:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = tmp.name
        tmp.close()
        try:
            df.to_csv(tmp_path, index=False, encoding=encoding)
        except Exception as e:
            logger.exception("Failed writing empty CSV to temp file: %s", e)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            st.error(f"Failed to prepare download: {e}")
            return
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = tmp.name
        tmp.close()
        try:
            df.to_csv(tmp_path, index=False, encoding=encoding)
        except Exception as e:
            logger.exception("Failed writing CSV to temp file: %s", e)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            st.error(f"Failed to prepare download: {e}")
            return

    try:
        f = open(tmp_path, "rb")
    except Exception as e:
        logger.exception("Failed to open temp file for download: %s", e)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        st.error(f"Failed to open generated file for download: {e}")
        return

    try:
        st.download_button(label=label, data=f, file_name=file_name, mime=mime)
    finally:
        try:
            f.close()
        except Exception:
            pass
        try:
            t = threading.Timer(max(5, int(tmp_life_seconds)), _safe_unlink, args=(tmp_path,))
            t.daemon = True
            t.start()
        except Exception:
            try:
                _safe_unlink(tmp_path)
            except Exception:
                pass

# ---------------------------
# Networking helpers
# ---------------------------
def make_session_with_retries(total_retries=4, backoff_factor=0.4, status_forcelist=(429,500,502,503,504)):
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["HEAD","GET","OPTIONS","POST","PUT","DELETE"]),
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
    if resp.status_code != 200:
        raise RuntimeError(f"Token API error {resp.status_code}: {resp.text[:1000]}")
    j = resp.json()
    token = j.get("access_token")
    expires_in = j.get("expires_in")
    expires_at = None
    if token and isinstance(expires_in, (int, float)):
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

def _raise_icis_error(resp: requests.Response, context: str):
    body = resp.text[:2000] if resp.text else ""
    raise RuntimeError(f"{context} failed with HTTP {resp.status_code}: {body}")

def fetch_spec_price_assessments(token: str) -> List[Dict[str, Any]]:
    sess = make_session_with_retries()
    sess.headers.update({"Accept":"application/json","Authorization":f"Bearer {token}"})
    resp = sess.get(SPEC_PRICEASSESSMENTS_ENDPOINT, timeout=30)
    if resp.status_code != 200:
        _raise_icis_error(resp, "Fetch price assessment specifications")
    return safe_parse_json(resp).get("data", [])

def fetch_price_assessments(token: str, specification_id: str, start_str: str, end_str: str) -> Dict[str, Any]:
    sess = make_session_with_retries()
    sess.headers.update({"Accept":"application/json","Authorization":f"Bearer {token}"})
    params = {
        "filter[specificationId]": specification_id,
        "filter[createdForDateStart]": start_str,  # YYYYMMDD required by ICIS
        "filter[createdForDateEnd]": end_str,      # YYYYMMDD required by ICIS
        "page[size]": PAGE_SIZE,
    }
    url = PRICE_ASSESSMENTS_ENDPOINT
    all_data = []
    all_included = []

    while url:
        resp = sess.get(url, params=params if params else None, timeout=60)
        logger.info("ICIS PriceAssessment URL: %s", resp.url)
        logger.info("ICIS PriceAssessment status: %s", resp.status_code)

        if resp.status_code != 200:
            _raise_icis_error(resp, "Fetch price assessments")
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
    if resp.status_code != 200:
        _raise_icis_error(resp, "Fetch trade specifications")
    return safe_parse_json(resp).get("data", [])

def fetch_trades(token: str, specification_ids: List[str], start_str: str, end_str: str) -> Dict[str, Any]:
    sess = make_session_with_retries()
    sess.headers.update({"Accept":"application/json","Authorization":f"Bearer {token}"})
    combined_data = []
    combined_included = []

    for spec_id in specification_ids:
        params = {
            "filter[specificationId]": spec_id,
            "filter[createdForDateStart]": start_str,  # YYYYMMDD required by ICIS
            "filter[createdForDateEnd]": end_str,      # YYYYMMDD required by ICIS
            "page[size]": PAGE_SIZE,
        }
        url = TRADES_ENDPOINT

        while url:
            resp = sess.get(url, params=params if params else None, timeout=60)
            logger.info("ICIS Trades URL: %s", resp.url)
            logger.info("ICIS Trades status: %s", resp.status_code)

            if resp.status_code != 200:
                _raise_icis_error(resp, f"Fetch trades for spec {spec_id}")
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
            "dataUsed": atts.get("dataUsed"),
            "isEstimate": atts.get("isEstimate"),
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
# Priority & date helpers
# ---------------------------
def get_priority_from_text(txt: Optional[str]) -> int:
    if not txt:
        return 999
    s = str(txt).upper()
    for k, v in PRIORITY_ORDER.items():
        if k in s:
            return v
    if "WKND" in s or "WEEKEND" in s:
        return 2
    if "DAY-AHEAD" in s or "DAY AHEAD" in s or (" DA" in s and len(s) <= 4) or s == "DA":
        return 1
    if "BOM" in s or "BALANCE" in s:
        return 4
    if "WDNW" in s or "WD/NW" in s:
        return 3
    if "MONTH" in s or s.startswith("M"):
        return 5
    if "QUARTER" in s or ("Q" in s and len(s) <= 3):
        return 6
    if "SEASON" in s or "WINTER" in s or "SUMMER" in s:
        return 7
    if "YEAR" in s:
        return 8
    return 999

def get_priority_from_row(row: pd.Series) -> int:
    code = row.get("period_relativePeriodCode") if "period_relativePeriodCode" in row.index else None
    label = row.get("period_relativePeriodLabel") if "period_relativePeriodLabel" in row.index else None
    default_label = row.get("period_defaultPeriodLabel") if "period_defaultPeriodLabel" in row.index else None
    p = 999
    if code:
        p = get_priority_from_text(code)
    if p == 999 and label:
        p = get_priority_from_text(label)
    if p == 999 and default_label:
        p = get_priority_from_text(default_label)
    return int(p if p is not None else 999)

def parse_date_flexible(x: Any) -> Optional[date]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    if isinstance(x, (date, datetime, pd.Timestamp)):
        try:
            return pd.to_datetime(x).date()
        except Exception:
            pass
    s = str(x)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ","%Y-%m-%dT%H:%M:%S.%fZ","%Y-%m-%d","%d/%m/%Y","%d-%m-%Y","%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        return dt.date()
    except Exception:
        return pd.NaT

# ---------------------------
# Per-day allocation & preserve-average
# ---------------------------
def build_daily_curve_allocation(df_assess: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["date","market_price","source_assessment_id","source_period_id","source_label","priority"]
    if df_assess is None or df_assess.empty:
        return pd.DataFrame(columns=out_cols)
    df = df_assess.copy()
    df["period_start_date"] = df.get("period_startDate").apply(parse_date_flexible) if "period_startDate" in df.columns else pd.NaT
    df["period_end_date"] = df.get("period_endDate").apply(parse_date_flexible) if "period_endDate" in df.columns else pd.NaT
    df["midpoint_num"] = pd.to_numeric(df.get("midpoint"), errors="coerce")
    dropped_mask = df["period_start_date"].isna() | df["period_end_date"].isna() | df["midpoint_num"].isna()
    if dropped_mask.any():
        dropped = df.loc[dropped_mask, ["assessment_id","period_startDate","period_endDate","midpoint"]].copy()
        try:
            st.warning(f"{len(dropped)} PriceAssessment rows could not be parsed and were ignored.")
            st.dataframe(dropped)
        except Exception:
            logger.warning("Dropped assessments: %s", dropped.to_dict("records"))
    df = df.dropna(subset=["period_start_date","period_end_date","midpoint_num"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=out_cols)
    df["priority"] = df.apply(get_priority_from_row, axis=1)
    df["period_len"] = (pd.to_datetime(df["period_end_date"]) - pd.to_datetime(df["period_start_date"])).dt.days + 1
    df["period_displayOrder_num"] = pd.to_numeric(df.get("period_displayOrder", pd.Series([pd.NA]*len(df))), errors="coerce").fillna(999999)
    df = df.sort_values(by=["priority","period_displayOrder_num","period_len"]).reset_index(drop=True)
    overall_start = df["period_start_date"].min()
    overall_end = df["period_end_date"].max()
    all_days = pd.date_range(overall_start, overall_end, freq="D").date
    out = pd.DataFrame({"date": all_days})
    out["market_price"] = pd.NA
    out["priority"] = pd.NA
    out["source_assessment_id"] = pd.NA
    out["source_period_id"] = pd.NA
    out["source_label"] = pd.NA

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
                existing_pr = out.at[idx,"priority"]
                if pd.isna(existing_pr) or int(existing_pr) > pr:
                    out.at[idx,"market_price"] = float(mid)
                    out.at[idx,"priority"] = pr
                    out.at[idx,"source_assessment_id"] = assessment_id
                    out.at[idx,"source_period_id"] = period_id
                    out.at[idx,"source_label"] = label
        except Exception as ex:
            logger.exception("Error applying assessment %s to dates %s..%s: %s", assessment_id, s, e, ex)
            continue
    out["market_price"] = pd.to_numeric(out["market_price"], errors="coerce")
    return out[["date","market_price","source_assessment_id","source_period_id","source_label","priority"]]

def assign_days_preserve_period_average(df_assess: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["date","market_price","source_assessment_id","source_period_id","source_label","priority","note"]
    if df_assess is None or df_assess.empty:
        return pd.DataFrame(columns=out_cols)
    df = df_assess.copy()
    df["period_start_date"] = df.get("period_startDate").apply(parse_date_flexible) if "period_startDate" in df.columns else pd.NaT
    df["period_end_date"] = df.get("period_endDate").apply(parse_date_flexible) if "period_endDate" in df.columns else pd.NaT
    df["midpoint_num"] = pd.to_numeric(df.get("midpoint"), errors="coerce")
    df = df.dropna(subset=["period_start_date","period_end_date","midpoint_num"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=out_cols)
    df["priority"] = df.apply(get_priority_from_row, axis=1)
    df["displayOrder_num"] = pd.to_numeric(df.get("period_displayOrder"), errors="coerce").fillna(999999)
    df["period_len"] = (pd.to_datetime(df["period_end_date"]) - pd.to_datetime(df["period_start_date"])).dt.days + 1
    df = df.sort_values(by=["priority","displayOrder_num","period_len"]).reset_index(drop=True)
    overall_start = df["period_start_date"].min()
    overall_end = df["period_end_date"].max()
    all_days = pd.date_range(overall_start, overall_end, freq="D").date
    out = pd.DataFrame({"date": all_days})
    out["market_price"] = np.nan
    out["priority"] = np.nan
    out["source_assessment_id"] = None
    out["source_period_id"] = None
    out["source_label"] = None
    out["note"] = ""

    for _, r in df.iterrows():
        s = r["period_start_date"]; e = r["period_end_date"]
        if pd.isna(s) or pd.isna(e):
            continue
        period_mask = (out["date"] >= s) & (out["date"] <= e)
        assigned_mask = period_mask & out["market_price"].notna()
        unassigned_mask = period_mask & out["market_price"].isna()
        assigned_prices = out.loc[assigned_mask, "market_price"].dropna().tolist()
        n_total = (e - s).days + 1
        n_assigned = len(assigned_prices)
        n_unassigned = n_total - n_assigned
        orig_mean = float(r["midpoint_num"])
        total_needed = orig_mean * n_total
        sum_assigned = sum(float(x) for x in assigned_prices) if assigned_prices else 0.0
        if n_unassigned <= 0:
            continue
        remaining_sum = total_needed - sum_assigned
        price_for_unassigned = remaining_sum / n_unassigned if n_unassigned > 0 else orig_mean
        if not np.isfinite(price_for_unassigned):
            price_for_unassigned = orig_mean
        out.loc[unassigned_mask, "market_price"] = float(price_for_unassigned)
        out.loc[unassigned_mask, "priority"] = int(r["priority"]) if not pd.isna(r["priority"]) else None
        out.loc[unassigned_mask, "source_assessment_id"] = r.get("assessment_id")
        label = r.get("period_relativePeriodLabel") or r.get("period_defaultPeriodLabel") or r.get("period_relativePeriodCode")
        out.loc[unassigned_mask, "source_label"] = label
        out.loc[unassigned_mask, "source_period_id"] = r.get("period_id")
        if n_assigned > 0:
            out.loc[unassigned_mask, "note"] = f"preserve_avg: orig_mean={orig_mean:.6f}, assigned_sum={sum_assigned:.6f}, remaining_sum={remaining_sum:.6f}, n_unassigned={n_unassigned}, per_day={price_for_unassigned:.6f}"
        else:
            out.loc[unassigned_mask, "note"] = f"direct_assign_midpoint={orig_mean:.6f}"
    out["market_price"] = pd.to_numeric(out["market_price"], errors="coerce")
    return out[["date","market_price","source_assessment_id","source_period_id","source_label","priority","note"]]

# ---------------------------
# Period-level aggregations and expansions
# ---------------------------
def compute_period_prices_from_assessments(df_assess: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if df_assess is None or df_assess.empty:
        return pd.DataFrame(columns=["period_id","period_price","period_startDate","period_endDate","period_label","n_assessments","mean_midpoint"])
    df = df_assess.copy()
    df["midpoint_num"] = pd.to_numeric(df["midpoint"], errors="coerce")

    def agg_group(g):
        out = {}
        out["period_startDate"] = g["period_startDate"].dropna().iloc[0] if g["period_startDate"].notna().any() else None
        out["period_endDate"] = g["period_endDate"].dropna().iloc[0] if g["period_endDate"].notna().any() else None
        out["period_label"] = (
            g["period_defaultPeriodLabel"].dropna().iloc[0]
            if "period_defaultPeriodLabel" in g and g["period_defaultPeriodLabel"].notna().any()
            else (
                g.get("period_relativePeriodLabel", pd.Series()).dropna().iloc[0]
                if "period_relativePeriodLabel" in g and g["period_relativePeriodLabel"].notna().any()
                else None
            )
        )
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
    return grouped[["period_id","period_price","period_startDate","period_endDate","period_label","n_assessments","mean_midpoint"]]

def expand_period_prices_to_daily(period_prices_df: pd.DataFrame, prefer_shortest_if_overlap: bool = True) -> pd.DataFrame:
    if period_prices_df is None or period_prices_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in period_prices_df.iterrows():
        try:
            s = parse_date_flexible(r.get("period_startDate"))
            e = parse_date_flexible(r.get("period_endDate"))
        except Exception:
            s = None; e = None
        if s is None or e is None or pd.isna(s) or pd.isna(e):
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
        daily = daily.sort_values(["date","period_len"]).drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily[["date","period_price","period_id","period_startDate","period_endDate","period_label"]]

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
    month_map: Dict[Tuple[int,int], Dict[str,Any]] = {}
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
    fill_from_broader(quarter,3)
    fill_from_broader(season,6)
    fill_from_broader(year,12)
    rows = []
    for (y,m), info in sorted(month_map.items()):
        rows.append({"year": y, "month": m, "period_price": info["price"], "source_period_id": info.get("source_period_id")})
    return pd.DataFrame(rows)

def fill_missing_days_from_periods(per_day_df: pd.DataFrame, period_prices_df: pd.DataFrame) -> pd.DataFrame:
    if per_day_df is None:
        return per_day_df
    out = per_day_df.copy()
    if "date" not in out.columns:
        return out
    if period_prices_df is None or period_prices_df.empty:
        return out
    pdaily = expand_period_prices_to_daily(period_prices_df, prefer_shortest_if_overlap=True)
    if pdaily is None or pdaily.empty:
        return out
    pdaily = pdaily.rename(columns={"period_price":"period_price_daily"})
    pdaily["date"] = pd.to_datetime(pdaily["date"]).dt.date
    map_series = pd.Series(pdaily["period_price_daily"].values, index=pdaily["date"].values).to_dict()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    if "filled_from_period" not in out.columns:
        out["filled_from_period"] = False
        out["filled_period_price"] = np.nan
    missing_idx = out[out["market_price"].isna()].index
    for idx in missing_idx:
        d = out.at[idx,"date"]
        if d in map_series and not pd.isna(map_series[d]):
            out.at[idx,"market_price"] = float(map_series[d])
            out.at[idx,"filled_from_period"] = True
            out.at[idx,"filled_period_price"] = float(map_series[d])
    return out

# ---------------------------
# WDNW & BOM helpers
# ---------------------------
def is_working_day(dt: date, country_code: str) -> bool:
    if not isinstance(dt, (date, datetime)):
        dt = pd.to_datetime(dt).date()
    if not _HAS_HOLIDAYS:
        return dt.weekday() < 5
    try:
        cal = holidays.CountryHoliday(country_code, years=[dt.year])
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
    return [dd for dd in weekend_days if isinstance(dd,(date,datetime))]

def next_week_working_days(report_date: date, country_code: str) -> List[date]:
    d = report_date
    days_ahead = (7 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_monday = d + timedelta(days=days_ahead)
    workdays=[]
    for i in range(7):
        day = next_monday + timedelta(days=i)
        if is_working_day(day,country_code) and day.weekday() < 5:
            workdays.append(day)
    return workdays[:5]

def derive_wdnw_days_and_price(report_date: date, alloc_curve: pd.DataFrame, assess_daily: pd.DataFrame, trades_daily: pd.DataFrame, country_code: str, wdnw_mode: str="Equal", da_weight: float=0.5) -> Tuple[List[date], Optional[float]]:
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
    priced_days = [d for d,p in day_prices.items() if p is not None]
    if not priced_days:
        return wdnw_days, None
    if wdnw_mode == "Equal":
        avg_price = float(np.mean([day_prices[d] for d in priced_days]))
        return wdnw_days, avg_price
    da_day = next_working_gas_day(report_date, country_code)
    try:
        da_day = pd.to_datetime(da_day).date()
    except Exception:
        pass
    if da_day not in priced_days:
        avg_price = float(np.mean([day_prices[d] for d in priced_days]))
        return wdnw_days, avg_price
    da_w = float(max(0.0, min(1.0, da_weight)))
    other_days = [d for d in priced_days if d != da_day]
    n_other = len(other_days)
    if n_other == 0:
        return wdnw_days, float(day_prices[da_day])
    other_total = 1.0 - da_w
    per_other = other_total / n_other
    weighted = da_w * day_prices[da_day] + sum(per_other * day_prices[d] for d in other_days)
    return wdnw_days, float(weighted)

def derive_bom_days_and_price(report_date: date, alloc_curve: pd.DataFrame, assess_daily: pd.DataFrame, trades_daily: pd.DataFrame, country_code: str) -> Tuple[List[date], Optional[float]]:
    da = next_working_gas_day(report_date, country_code)
    wknd = next_weekend(report_date, country_code)
    wknd_start = min(wknd) if wknd else da
    earlier = min(pd.to_datetime(da).date(), pd.to_datetime(wknd_start).date())
    bom_start = earlier + timedelta(days=1)
    if bom_start.month != report_date.month:
        bom_start = (pd.Timestamp(report_date.replace(day=1)) + pd.DateOffset(months=1)).date()
    month_end = (pd.Timestamp(bom_start).replace(day=1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date()
    days = pd.date_range(start=bom_start, end=month_end, freq="D").date
    prices=[]
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
# Trades / MTM helpers
# ---------------------------
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

def expand_trade_to_daily(trade_row: Dict[str,Any], assume_side: str="BUY", hub_country: Optional[str]="GB") -> pd.DataFrame:
    start_raw = trade_row.get("period_startDate")
    end_raw = trade_row.get("period_endDate")
    trade_dt_raw = trade_row.get("tradeDateTime")
    tz_name = HUB_TIMEZONES.get(hub_country, "Europe/Berlin")
    gas_day_start_hour = GAS_DAY_START_OVERRIDES.get(hub_country, DEFAULT_GAS_DAY_START)
    try:
        if start_raw and end_raw:
            start = parse_date_flexible(start_raw)
            end = parse_date_flexible(end_raw)
        else:
            if trade_dt_raw:
                gd = timestamp_to_gas_day(trade_dt_raw, tz_name, gas_day_start_hour)
                start = gd; end = gd
            else:
                start = parse_date_flexible(trade_row.get("createdForDate") or datetime.utcnow().date())
                end = start
    except Exception:
        start = parse_date_flexible(trade_row.get("createdForDate") or datetime.utcnow().date())
        end = start
    days = pd.date_range(start=start, end=end, freq="D").date if start and end else [parse_date_flexible(trade_row.get("createdForDate") or datetime.utcnow().date())]
    n_days = len(days) if len(days)>0 else 1
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
    rows=[]
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
        r.update(period_fields); r.update(ts_fields)
        rows.append(r)
    return pd.DataFrame(rows)

def compute_period_prices_from_trades(trades_df: pd.DataFrame, method: str="VWAP", trim_pct: float=0.1) -> pd.DataFrame:
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
    rows=[]
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
            high = priced["price_num"].quantile(1-trim_pct)
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

def compute_mtm_from_trades(trades_df: pd.DataFrame, valuation_curve: pd.DataFrame, trade_filter: Optional[List[str]] = None, multiplier: float = 1.0) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()
    df_trades = trades_df.copy()
    if trade_filter:
        sel_set = {str(x) for x in trade_filter}
        df_trades = df_trades[df_trades["trade_id"].astype(str).isin(sel_set)].copy()
        if df_trades.empty:
            return pd.DataFrame()
    daily_rows=[]
    for _, tr in df_trades.iterrows():
        try:
            edf = expand_trade_to_daily(tr)
        except Exception:
            logger.exception("expand_trade_to_daily failed for trade %s", tr.get("trade_id"))
            continue
        if edf is None or edf.empty:
            continue
        edf["trade_id"] = tr.get("trade_id")
        edf["trade_price"] = pd.to_numeric(tr.get("price"), errors="coerce")
        edf["trade_side"] = tr.get("tradeSide") or "BUY"
        edf["period_id"] = tr.get("period_id")
        edf["period_startDate"] = tr.get("period_startDate")
        edf["period_endDate"] = tr.get("period_endDate")
        for lbl_col in ("period_relativePeriodLabel","period_defaultPeriodLabel","period_absolutePeriodCode","period_relativePeriodCode"):
            if lbl_col in tr:
                edf[lbl_col] = tr.get(lbl_col)
        daily_rows.append(edf)
    if not daily_rows:
        return pd.DataFrame()
    trades_daily = pd.concat(daily_rows, ignore_index=True)
    trades_daily["date"] = pd.to_datetime(trades_daily["date"]).dt.date
    val = valuation_curve.copy()
    val["date"] = pd.to_datetime(val["date"]).dt.date
    if "market_price" not in val.columns and "period_price" in val.columns:
        val = val.rename(columns={"period_price":"market_price"})
    val_map = pd.Series(val["market_price"].values, index=val["date"].values).to_dict()

    def _row_pnl(row):
        d = row["date"]
        market_price = val_map.get(d, np.nan)
        row_price = float(row.get("trade_price") if pd.notna(row.get("trade_price")) else np.nan)
        vol = row.get("daily_volume") if row.get("daily_volume") is not None else 0.0
        try:
            vol = float(vol)
        except Exception:
            vol = 0.0
        side = str(row.get("trade_side") or "BUY").upper()
        if pd.isna(market_price) or pd.isna(row_price):
            return pd.Series({"market_price": market_price, "pnl": np.nan})
        if side.startswith("B"):
            pnl = (market_price - row_price) * vol * float(multiplier)
        else:
            pnl = (row_price - market_price) * vol * float(multiplier)
        return pd.Series({"market_price": market_price, "pnl": pnl})

    md = trades_daily.apply(_row_pnl, axis=1)
    trades_daily = pd.concat([trades_daily.reset_index(drop=True), md.reset_index(drop=True)], axis=1)
    trades_daily["pnl"] = pd.to_numeric(trades_daily["pnl"], errors="coerce")
    trades_daily["market_price"] = pd.to_numeric(trades_daily["market_price"], errors="coerce")
    return trades_daily

# ---------- MTM aggregation & analytics ----------
def aggregate_portfolio_mtm(trades_daily: pd.DataFrame) -> pd.DataFrame:
    if trades_daily is None or trades_daily.empty:
        return pd.DataFrame(columns=["date","pnl_total","pnl_count","mean_market","mean_trade","cum_pnl"])
    agg = trades_daily.groupby("date", dropna=False).agg(
        pnl_total=("pnl","sum"),
        pnl_count=("pnl", lambda x: x.notna().sum()),
        mean_market=("market_price","mean"),
        mean_trade=("trade_price","mean")
    ).reset_index()
    agg = agg.sort_values("date")
    agg["cum_pnl"] = agg["pnl_total"].cumsum()
    return agg

def top_contributors_by_period(trades_daily: pd.DataFrame, top_n: int=10) -> pd.DataFrame:
    if trades_daily is None or trades_daily.empty:
        return pd.DataFrame(columns=["period_key","period_relative_label","period_absolute_code","pnl_sum","abs_pnl_sum","n_trades","n_days"])
    df = trades_daily.copy()

    def _relative_label(row):
        for col in ("period_relativePeriodLabel","period_defaultPeriodLabel","period_relativePeriodCode"):
            if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                return str(row[col])
        return ""
    df["period_relative_label"] = df.apply(_relative_label, axis=1)
    df["period_absolute_code"] = df.get("period_absolutePeriodCode", pd.Series([""] * len(df))).fillna("").astype(str)

    def _make_key(row):
        lab = row.get("period_relative_label","") or ""
        code = row.get("period_absolute_code","") or ""
        if lab and code:
            return f"{lab} | {code}"
        if lab:
            return lab
        if code:
            return code
        return str(row.get("trade_id"))
    df["period_key"] = df.apply(_make_key, axis=1)

    grouped = df.groupby(["period_key","period_relative_label","period_absolute_code"], dropna=False).agg(
        pnl_sum=("pnl","sum"),
        abs_pnl_sum=("pnl", lambda s: float(np.nansum(np.abs(s)))),
        n_trades=("trade_id", lambda s: s.astype(str).nunique()),
        n_days=("date","nunique")
    ).reset_index()
    grouped = grouped.sort_values("abs_pnl_sum", ascending=False).head(top_n).reset_index(drop=True)
    grouped = grouped[["period_key","period_relative_label","period_absolute_code","pnl_sum","abs_pnl_sum","n_trades","n_days"]]
    return grouped

# ---------- Robust VaR helpers ----------
def compute_parametric_var(spread_series: Any, trade_size: float, z: float=1.645) -> float:
    try:
        if spread_series is None:
            return float("nan")
        if not isinstance(spread_series, (pd.Series, np.ndarray, list)):
            spread_series = pd.Series([spread_series])
        s = pd.Series(spread_series).dropna()
        if s.empty:
            return float("nan")
        s = s.astype(float, errors="ignore")
        sigma = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        return float(abs(z * sigma * float(trade_size)))
    except Exception as e:
        logger.exception("compute_parametric_var failed: %s", e)
        return float("nan")

def compute_historical_var(spread_series: Any, trade_size: float, alpha: float=0.95) -> float:
    try:
        if spread_series is None:
            return float("nan")
        if not isinstance(spread_series, (pd.Series, np.ndarray, list)):
            spread_series = pd.Series([spread_series])
        s = pd.Series(spread_series).dropna()
        if s.empty:
            return float("nan")
        s = s.astype(float, errors="ignore")
        diffs = s - s.mean()
        q = np.quantile(diffs, 1 - float(alpha))
        loss = abs(float(q)) * float(trade_size)
        return float(loss)
    except Exception as e:
        logger.exception("compute_historical_var failed: %s", e)
        return float("nan")

# ============================================================
# IMPLIED VOLATILITY & VOLATILITY ANALYSIS HELPERS
# ============================================================
def compute_bid_ask_spread(trades_df: pd.DataFrame, period_id: str = None) -> Dict[str, Any]:
    if trades_df is None or trades_df.empty:
        return {"bid": np.nan, "ask": np.nan, "mid": np.nan, "spread": np.nan, "spread_pct": np.nan}

    df = trades_df.copy()
    df["price_num"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["side"] = df.get("tradeSide", "").astype(str).str.upper()
    if period_id is not None:
        df = df[df.get("period_id") == period_id].copy()

    sell_prices = df[df["side"].str.contains("SELL", na=False)]["price_num"].dropna()
    buy_prices = df[df["side"].str.contains("BUY", na=False)]["price_num"].dropna()

    bid = float(sell_prices.mean()) if not sell_prices.empty else np.nan
    ask = float(buy_prices.mean()) if not buy_prices.empty else np.nan
    mid = (bid + ask) / 2.0 if (not np.isnan(bid) and not np.isnan(ask)) else np.nan
    spread = (ask - bid) if (not np.isnan(bid) and not np.isnan(ask)) else np.nan
    spread_pct = (spread / mid * 100.0) if (not np.isnan(spread) and mid > 0) else np.nan

    return {"bid": bid, "ask": ask, "mid": mid, "spread": spread, "spread_pct": spread_pct}

def compute_price_volatility_by_period(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["period_id","period_label","n_trades","price_std","price_cv","log_return_vol","realized_vol"])

    df = trades_df.copy()
    df["price_num"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["tradeDateTime"] = pd.to_datetime(df.get("tradeDateTime"), errors="coerce")

    rows = []
    for pid, group in df.groupby("period_id", dropna=False):
        priced = group[group["price_num"].notna()].copy()
        if priced.empty or len(priced) < 2:
            continue

        prices = priced["price_num"].values
        n_trades = len(prices)

        price_std = float(np.std(prices, ddof=1)) if n_trades > 1 else 0.0
        price_mean = float(np.mean(prices))
        price_cv = (price_std / price_mean * 100.0) if price_mean > 0 else 0.0

        log_ret_vol = np.nan
        if "tradeDateTime" in priced.columns:
            sorted_p = priced.sort_values("tradeDateTime").copy()
            sorted_prices = sorted_p["price_num"].values
            if len(sorted_prices) > 1:
                log_returns = np.diff(np.log(sorted_prices))
                log_ret_vol = float(np.std(log_returns, ddof=1)) * 100.0

        price_range = prices.max() - prices.min()
        mid_price = (prices.max() + prices.min()) / 2.0
        realized_vol = (price_range / mid_price * 100.0) if mid_price > 0 else 0.0

        label = group.iloc[0].get("period_defaultPeriodLabel") or group.iloc[0].get("period_relativePeriodLabel") or str(pid)

        rows.append({
            "period_id": pid,
            "period_label": label,
            "n_trades": n_trades,
            "price_std": price_std,
            "price_cv": price_cv,
            "log_return_vol": log_ret_vol,
            "realized_vol": realized_vol
        })

    return pd.DataFrame(rows)

def compute_volume_weighted_volatility(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["period_id","period_label","vwap","volume_weighted_vol","total_volume"])

    df = trades_df.copy()
    df["price_num"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["volume_num"] = pd.to_numeric(df.get("volume"), errors="coerce")

    rows = []
    for pid, group in df.groupby("period_id", dropna=False):
        priced = group[(group["price_num"].notna()) & (group["volume_num"].notna())].copy()
        if priced.empty:
            continue

        prices = priced["price_num"].values
        volumes = priced["volume_num"].values

        total_vol = float(volumes.sum())
        vwap = float((prices * volumes).sum() / total_vol) if total_vol > 0 else np.nan

        if not np.isnan(vwap) and total_vol > 0:
            vwv = float(((prices - vwap) ** 2 * volumes).sum() / total_vol)
            vwv_std = np.sqrt(vwv) if vwv > 0 else 0.0
            vw_vol_pct = (vwv_std / vwap * 100.0) if vwap > 0 else 0.0
        else:
            vw_vol_pct = np.nan

        label = group.iloc[0].get("period_defaultPeriodLabel") or group.iloc[0].get("period_relativePeriodLabel") or str(pid)

        rows.append({
            "period_id": pid,
            "period_label": label,
            "vwap": vwap,
            "volume_weighted_vol": vw_vol_pct,
            "total_volume": total_vol
        })

    return pd.DataFrame(rows)

def compute_term_structure_vol(period_prices_df: pd.DataFrame) -> pd.DataFrame:
    if period_prices_df is None or period_prices_df.empty:
        return pd.DataFrame(columns=["date","ts_vol","n_periods"])

    df = period_prices_df.copy()
    df["period_price_num"] = pd.to_numeric(df.get("period_price"), errors="coerce")

    rows = []
    for d, group in df.groupby("date", dropna=False):
        prices = group["period_price_num"].dropna().values
        if len(prices) < 2:
            continue
        spreads = np.diff(prices)
        ts_vol = float(np.std(spreads, ddof=1)) if len(spreads) > 1 else 0.0
        rows.append({"date": d, "ts_vol": ts_vol, "n_periods": len(prices)})

    return pd.DataFrame(rows)

def compute_moneyness_and_iv_proxy(trades_df: pd.DataFrame, assessment_curve: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    df["price_num"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["period_start"] = pd.to_datetime(df.get("period_startDate"), errors="coerce").dt.date
    df["period_end"] = pd.to_datetime(df.get("period_endDate"), errors="coerce").dt.date

    assess_map = {}
    if assessment_curve is not None and not assessment_curve.empty:
        a = assessment_curve.copy()
        for _, row in a.iterrows():
            pid = row.get("period_id")
            price = row.get("period_price")
            if pid and not pd.isna(price):
                assess_map[pid] = float(price)

    rows = []
    for pid, group in df.groupby("period_id", dropna=False):
        priced = group[group["price_num"].notna()].copy()
        if priced.empty:
            continue

        spread_info = compute_bid_ask_spread(priced)

        assess_price = assess_map.get(pid, np.nan)
        prices = priced["price_num"].values
        if not np.isnan(assess_price) and assess_price > 0:
            moneyness = (prices / assess_price - 1.0) * 100.0
        else:
            moneyness = np.array([np.nan] * len(prices))

        volumes = pd.to_numeric(priced.get("volume"), errors="coerce").values
        if volumes.sum() > 0:
            vol_shares = volumes / volumes.sum()
            hhi = float((vol_shares ** 2).sum())
        else:
            hhi = np.nan

        spread_pct = spread_info.get("spread_pct", np.nan)
        moneyness_std = float(np.nanstd(moneyness, ddof=1)) if len(moneyness) > 1 else np.nan

        if not np.isnan(spread_pct):
            iv_proxy = spread_pct * (1.0 + 0.5 * hhi) if not np.isnan(hhi) else spread_pct
        else:
            iv_proxy = moneyness_std

        label = group.iloc[0].get("period_defaultPeriodLabel") or group.iloc[0].get("period_relativePeriodLabel") or str(pid)

        rows.append({
            "period_id": pid,
            "period_label": label,
            "n_trades": len(priced),
            "bid_price": spread_info["bid"],
            "ask_price": spread_info["ask"],
            "mid_price": spread_info["mid"],
            "spread_pct": spread_info["spread_pct"],
            "assess_price": assess_price,
            "moneyness_mean": float(np.nanmean(moneyness)),
            "moneyness_std": moneyness_std,
            "herfindahl_index": hhi,
            "iv_proxy_pct": iv_proxy
        })

    return pd.DataFrame(rows)

def compute_rolling_curve_volatility(daily_curve_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if daily_curve_df is None or daily_curve_df.empty:
        return pd.DataFrame(columns=["date","rolling_curve_vol","n_obs"])

    df = daily_curve_df.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.date
    df["price"] = pd.to_numeric(df.get("market_price") or df.get("period_price"), errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    df["price_returns"] = np.log(df["price"] / df["price"].shift(1)) * 100.0

    rows = []
    for i in range(window, len(df)):
        window_returns = df.iloc[i-window:i]["price_returns"].dropna().values
        if len(window_returns) > 0:
            rolling_vol = float(np.std(window_returns, ddof=1))
            rows.append({
                "date": df.iloc[i]["date"],
                "rolling_curve_vol": rolling_vol,
                "n_obs": len(window_returns)
            })

    return pd.DataFrame(rows)

# ============================================================
# LOCATION SPREAD ANALYSIS
# ============================================================
def build_location_spread_analysis(
    token: str,
    selected_specs: Dict[str, str],
    base_label: str,
    start_str: str,
    end_str: str,
    assess_method: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Calculate spreads only where the comparison hub and reference hub have
    the same delivery-period start and end dates.

    Spread = comparison hub price - reference hub price.
    """
    skipped: List[str] = []
    period_prices_by_hub: Dict[str, pd.DataFrame] = {}

    for hub_label, specification_id in selected_specs.items():
        combined = fetch_price_assessments(token, specification_id, start_str, end_str)
        assessments = parse_assessments_with_periods(combined)

        if assessments.empty:
            skipped.append(hub_label)
            continue

        periods = compute_period_prices_from_assessments(
            assessments, method=assess_method
        )
        if periods.empty:
            skipped.append(hub_label)
            continue

        periods = periods.copy()
        periods["period_start"] = periods["period_startDate"].apply(parse_date_flexible)
        periods["period_end"] = periods["period_endDate"].apply(parse_date_flexible)
        periods["period_price"] = pd.to_numeric(periods["period_price"], errors="coerce")
        periods = periods.dropna(subset=["period_start","period_end","period_price"])
        periods = periods.drop_duplicates(subset=["period_start", "period_end"], keep="last")

        if periods.empty:
            skipped.append(hub_label)
            continue

        period_prices_by_hub[hub_label] = periods[
            ["period_start","period_end","period_label","period_price"]
        ].copy()

    if base_label not in period_prices_by_hub:
        return pd.DataFrame(), pd.DataFrame(), skipped

    reference = period_prices_by_hub[base_label].rename(
        columns={
            "period_label": "reference_period_label",
            "period_price": "reference_price",
        }
    )

    spread_frames = []

    for hub_label, comparison in period_prices_by_hub.items():
        if hub_label == base_label:
            continue

        comparison = comparison.rename(
            columns={
                "period_label": "comparison_period_label",
                "period_price": "comparison_price",
            }
        )

        matched = reference.merge(
            comparison,
            on=["period_start", "period_end"],
            how="inner",
            validate="one_to_one",
        )

        if matched.empty:
            continue

        matched["comparison_hub"] = hub_label
        matched["reference_hub"] = base_label
        matched["spread"] = matched["comparison_price"] - matched["reference_price"]
        matched["period_label"] = matched["comparison_period_label"].fillna(matched["reference_period_label"])
        spread_frames.append(matched)

    if not spread_frames:
        return pd.DataFrame(), pd.DataFrame(), skipped

    spreads = pd.concat(spread_frames, ignore_index=True)
    spreads = spreads.sort_values(["comparison_hub","period_start","period_end"]).reset_index(drop=True)

    summary_rows = []
    for hub_label, group in spreads.groupby("comparison_hub"):
        group = group.sort_values(["period_start","period_end"])
        nearest = group.iloc[0]
        spread_value = float(nearest["spread"])

        if spread_value > 0:
            position = f"{hub_label} is at a premium to {base_label}"
        elif spread_value < 0:
            position = f"{hub_label} is at a discount to {base_label}"
        else:
            position = f"{hub_label} is at parity with {base_label}"

        summary_rows.append({
            "comparison_hub": hub_label,
            "reference_hub": base_label,
            "matched_delivery_periods": len(group),
            "nearest_period": nearest["period_label"],
            "nearest_period_spread": spread_value,
            "mean_spread_across_matched_periods": float(group["spread"].mean()),
            "interpretation": (
                f"For the nearest matched delivery period, {position}. "
                "Positive values indicate a comparison-hub premium; negative values "
                "indicate a discount. Only matching delivery periods are included."
            ),
        })

    return spreads, pd.DataFrame(summary_rows), skipped

# ---------------------------
# Streamlit UI wiring
# ---------------------------
st.set_page_config(page_title="ICIS Forward Gas Curves — MTM & IV Insights", layout="wide")
st.title("ICIS Forward Gas Curves — MTM & Implied Volatility Analysis")

# Session init
if "token" not in st.session_state: st.session_state["token"] = None
if "token_expires_at" not in st.session_state: st.session_state["token_expires_at"] = None
if "combined_pa_raw" not in st.session_state: st.session_state["combined_pa_raw"] = None

def token_valid() -> bool:
    t = st.session_state.get("token")
    if not t:
        return False
    exp = st.session_state.get("token_expires_at")
    if not exp:
        return True
    return datetime.utcnow() < (exp - timedelta(seconds=30))

# Sidebar: authentication & options
st.sidebar.header("Authentication")
client_id = st.sidebar.text_input("Client ID", type="password")
client_secret = st.sidebar.text_input("Client Secret", type="password")
paste_token = st.sidebar.text_area("Paste access_token (optional)")

if token_valid():
    st.sidebar.success("Token available")
    if st.sidebar.button("Logout and clear token"):
        st.session_state["token"] = None
        st.session_state["token_expires_at"] = None
        rerun_app()
else:
    if st.sidebar.button("Authenticate (client_id/secret)"):
        if not client_id or not client_secret:
            st.sidebar.error("Provide client_id and client_secret")
        else:
            try:
                tok = get_token_and_expiry(client_id, client_secret)
            except Exception as e:
                st.sidebar.error(f"Auth failed: {e}")
            else:
                st.session_state["token"] = tok["access_token"]
                st.session_state["token_expires_at"] = tok["expires_at"]
                st.sidebar.success("Authenticated")
                rerun_app()

    if st.sidebar.button("Use pasted token"):
        if not paste_token:
            st.sidebar.error("Paste token first")
        else:
            st.session_state["token"] = paste_token.strip()
            st.session_state["token_expires_at"] = None
            st.sidebar.success("Token stored")
            rerun_app()

if not token_valid():
    st.info("Authenticate in the sidebar to continue.")
    st.stop()

token = st.session_state["token"]
session = make_session_with_retries()
session.headers.update({"Authorization": f"Bearer {token}"})

# Load specs (cached)
if "price_specs" not in st.session_state:
    with st.spinner("Loading specifications..."):
        try:
            price_specs = fetch_spec_price_assessments(token)
            trade_specs = fetch_trade_specifications(token)
            st.session_state["price_specs"] = price_specs
            st.session_state["trade_specs"] = trade_specs
        except Exception as e:
            st.error(f"Failed to load specifications: {e}")
            st.stop()

# Build spec maps
price_spec_map = {s.get("id"): s.get("attributes", {}).get("name", f"{s.get('id')}") for s in st.session_state["price_specs"]}
price_spec_list = list(price_spec_map.values())
trade_spec_map = {s.get("id"): s.get("attributes", {}).get("name", f"{s.get('id')}") for s in st.session_state["trade_specs"]}
trade_display_by_id = {tid: f"{trade_spec_map[tid]} | id={tid}" for tid in trade_spec_map.keys()}
trade_display_list = [trade_display_by_id[tid] for tid in trade_display_by_id.keys()]
display_to_id = {trade_display_by_id[tid]: tid for tid in trade_display_by_id.keys()}

# Sidebar selectors
st.sidebar.header("Selectors & Options")
selected_price_spec_name = st.sidebar.selectbox("PriceAssessment specification (for curve & history)", options=[""] + price_spec_list)
selected_trade_spec_displays = st.sidebar.multiselect("Trade specification(s) (for trades)", options=trade_display_list)

st.sidebar.header("Location spread")
location_spec_options = {f"{name} | id={spec_id}": str(spec_id) for spec_id, name in price_spec_map.items()}
location_spec_labels = list(location_spec_options.keys())
selected_location_labels = st.sidebar.multiselect("Hub specifications to compare", options=location_spec_labels, default=[])
if selected_location_labels:
    location_base_label = st.sidebar.selectbox("Reference hub", options=selected_location_labels)
else:
    location_base_label = None

# Date range
st.sidebar.markdown("createdForDate range (YYYYMMDD required by ICIS)")
start_date = st.sidebar.date_input("createdForDateStart", value=(datetime.utcnow().date() - timedelta(days=90)))
end_date = st.sidebar.date_input("createdForDateEnd", value=datetime.utcnow().date())
start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

# Sidebar: MTM & curve options
st.sidebar.header("MTM & Insights")
sidebar_mtm_multiplier = st.sidebar.number_input("Unit multiplier / notional (per MWh)", value=1.0, step=0.1)
st.session_state["mtm_multiplier"] = float(sidebar_mtm_multiplier)
mtm_roll_days = st.sidebar.slider("Rolling vol window (days) for insights", min_value=5, max_value=90, value=20)
mtm_z = st.sidebar.slider("Parametric VaR z-score (e.g., 1.645 for 95%)", min_value=1.0, max_value=3.0, value=1.645, step=0.005)

st.sidebar.header("Curve processing options")
assess_method = st.sidebar.selectbox("Assessments aggregation", options=["mean","median","latest"], index=0)
trades_method = st.sidebar.selectbox("Trades aggregation", options=["VWAP","average","median","trimmed"], index=0)
trim_pct = st.sidebar.slider("Trim % (for trimmed mean)", min_value=0.0, max_value=0.25, value=0.1, step=0.01)
enforce_arbitrage = st.sidebar.checkbox("Enforce arbitrage-free monthly derivation", value=True)
preserve_avg_toggle = st.sidebar.checkbox("Preserve period-average when partially overlaid", value=True)
use_preserve_downstream = st.sidebar.checkbox("Use preserve-average per-day assignments for downstream valuation", value=True)

# ---------------------------
# Location spread analysis
# ---------------------------
st.header("Location spreads")

st.markdown("""
A location spread is:
    comparison hub price - reference hub price

Positive values mean the comparison hub is at a premium. Negative values mean
it is at a discount. A wider spread means the price difference is growing; a
narrower spread means the hubs are converging. This is a market spread, not
a forecast. The spread can reflect transport costs, pipeline constraints,
local supply-demand balance, storage, and other conditions.
""")

if st.button("Build location spreads", key="build_location_spreads"):
    if not selected_location_labels:
        st.error("Select at least two hub specifications to compare.")
    elif location_base_label is None or location_base_label not in selected_location_labels:
        st.error("Choose a reference hub from the selected hub specifications.")
    else:
        selected_location_specs = {label: location_spec_options[label] for label in selected_location_labels}
        with st.spinner("Fetching hub assessments and calculating spreads..."):
            try:
                location_spreads, location_summary, location_skipped_hubs = build_location_spread_analysis(
                    token=token,
                    selected_specs=selected_location_specs,
                    base_label=location_base_label,
                    start_str=start_str,
                    end_str=end_str,
                    assess_method=assess_method,
                )
                st.session_state["location_spreads"] = location_spreads
                st.session_state["location_summary"] = location_summary
                st.session_state["location_skipped_hubs"] = location_skipped_hubs
                st.session_state["location_base_label"] = location_base_label
            except Exception as e:
                logger.exception("Location spread calculation failed: %s", e)
                st.error(f"Could not build location spreads: {e}")

location_spreads = st.session_state.get("location_spreads", pd.DataFrame())
location_summary = st.session_state.get("location_summary", pd.DataFrame())
location_skipped_hubs = st.session_state.get("location_skipped_hubs", [])

if location_skipped_hubs:
    st.warning("No usable assessment periods were found for: " + ", ".join(location_skipped_hubs))

if not location_summary.empty:
    st.subheader("Latest location spread insights")
    st.dataframe(location_summary, use_container_width=True)

    fig_location = px.line(
        location_spreads,
        x="period_start",
        y="spread",
        color="comparison_hub",
        markers=True,
        hover_data=["period_label","period_end","comparison_price","reference_price"],
        title="Location spreads by matched delivery period",
        labels={
            "period_start":"Delivery period start",
            "spread":"Price difference (comparison - reference)",
            "comparison_hub":"Comparison hub"
        }
    )
    fig_location.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_location.update_layout(
        xaxis_title="Delivery period start",
        yaxis_title="Price difference (comparison hub − reference hub)",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig_location, use_container_width=True)

    st.subheader("Matched delivery periods")
    st.dataframe(location_spreads, use_container_width=True)
    st_download_df("Download location spreads CSV", location_spreads, file_name="location_spreads.csv", mime="text/csv")
else:
    st.info("Select at least two hub specifications, choose a reference hub, and click Build location spreads to calculate the spread curve.")

# ---------------------------
# Build PriceAssessment curve
# ---------------------------
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

        st.session_state["combined_pa_raw"] = combined_pa
        df_assess = parse_assessments_with_periods(combined_pa)
        st.session_state["df_assess"] = df_assess

        st.subheader("PriceAssessments (parsed rows)")
        if df_assess is None or df_assess.empty:
            st.warning("No PriceAssessment rows parsed. Inspect raw API response in 'Raw API response' expandable below.")
            with st.expander("Raw API response (first pages)"):
                st.json(combined_pa)
        else:
            st.dataframe(df_assess.head(500))
            st_download_df("Download parsed PriceAssessments CSV", df_assess, file_name="assessments_parsed.csv", mime="text/csv")

        alloc_daily = build_daily_curve_allocation(df_assess)
        st.session_state["alloc_curve"] = alloc_daily
        simple_per_day = alloc_daily.copy()
        preserve_per_day = assign_days_preserve_period_average(df_assess) if preserve_avg_toggle else simple_per_day.copy()

        period_prices_assess = compute_period_prices_from_assessments(df_assess, method=assess_method)

        if enforce_arbitrage:
            af_months = enforce_arbitrage_free_monthly(period_prices_assess)
            perperiod_daily = expand_period_prices_to_daily(af_months.rename(columns={"period_price":"period_price"}))
            if "period_price" in perperiod_daily.columns:
                perperiod_daily = perperiod_daily.rename(columns={"period_price":"market_price"})
            st.session_state["perperiod_daily"] = perperiod_daily
        else:
            perperiod_daily = expand_period_prices_to_daily(period_prices_assess)
            if "period_price" in perperiod_daily.columns:
                perperiod_daily = perperiod_daily.rename(columns={"period_price":"market_price"})
            st.session_state["perperiod_daily"] = perperiod_daily

        simple_per_day_filled = fill_missing_days_from_periods(simple_per_day, period_prices_assess)
        preserve_per_day_filled = fill_missing_days_from_periods(preserve_per_day, period_prices_assess)

        st.session_state["simple_per_day"] = simple_per_day_filled
        st.session_state["preserve_per_day"] = preserve_per_day_filled
        st.session_state["period_prices_assess"] = period_prices_assess
        st.session_state["assess_daily"] = preserve_per_day_filled if use_preserve_downstream else simple_per_day_filled

        st.success("Curves built and stored in session. Use the Fetch trades button to load trades and compute MTM.")

# Ensure assess_daily exists
if "assess_daily" not in st.session_state:
    st.info("Build a PriceAssessment curve (sidebar) to proceed with previews, trades and MTM.")
    st.stop()

# ---------------------------
# Trades fetch section (explicit button)
# ---------------------------
st.header("Trades: fetch and compute trades-derived curve")
st.write("Select trade specification(s) in the sidebar, then click Fetch trades.")
if st.button("Fetch trades"):
    if not selected_trade_spec_displays:
        st.error("Select at least one Trade Specification in the sidebar before fetching trades.")
    else:
        spec_ids = [display_to_id[disp] for disp in selected_trade_spec_displays if disp in display_to_id]
        if not spec_ids:
            st.error("Failed to determine tradeSpecification ids from selection.")
        else:
            with st.spinner("Fetching trades from API..."):
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
            else:
                period_prices_trades = compute_period_prices_from_trades(trades_df, method=trades_method if trades_method != "trimmed" else "trimmed", trim_pct=trim_pct)
                trades_derived_daily = expand_period_prices_to_daily(period_prices_trades)
                if "period_price" in trades_derived_daily.columns:
                    trades_derived_daily = trades_derived_daily.rename(columns={"period_price":"market_price"})
                st.session_state["trades_derived_daily"] = trades_derived_daily
                st.subheader("Trades-derived daily (sample)")
                st.dataframe(trades_derived_daily.head(200))
                st.success("Trades fetched and trades-derived curve computed and stored in session.")

# ---------------------------
# Curve preview & overlay
# ---------------------------
st.subheader("Curve preview & overlay")
alloc_norm = st.session_state.get("alloc_curve", pd.DataFrame())
assess_norm = st.session_state.get("assess_daily", pd.DataFrame())
trades_norm = st.session_state.get("perperiod_daily", pd.DataFrame())

def norm(df, col):
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", col])
    out = df[["date","market_price"]].copy() if "market_price" in df.columns else df[["date","period_price"]].rename(columns={"period_price":"market_price"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.rename(columns={"market_price": col})
    return out

alloc_plot = norm(alloc_norm, "allocation_price")
assess_plot = norm(assess_norm, "assess_price")
trades_plot = norm(trades_norm, "trades_price")

date_collections = [ddf["date"] for ddf in (alloc_plot, assess_plot, trades_plot) if isinstance(ddf, pd.DataFrame) and not ddf.empty]
if date_collections:
    concat_dates = pd.concat(date_collections)
    min_date = concat_dates.min()
    max_date = concat_dates.max()
    all_dates = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D").date})
else:
    all_dates = pd.DataFrame({"date": []})

combined = all_dates.merge(alloc_plot, on="date", how="left").merge(assess_plot, on="date", how="left").merge(trades_plot, on="date", how="left")
st.dataframe(combined.head(200))
st_download_df("Download combined curve CSV", combined, file_name="combined_curves.csv", mime="text/csv")

fig = go.Figure()
if not alloc_plot.empty:
    fig.add_trace(go.Scatter(x=pd.to_datetime(alloc_plot["date"]), y=alloc_plot["allocation_price"], name="Allocation (priority)", line=dict(color="gray")))
if not assess_plot.empty:
    fig.add_trace(go.Scatter(x=pd.to_datetime(assess_plot["date"]), y=assess_plot["assess_price"], name="Assessments (chosen)", line=dict(color="blue")))
if not trades_plot.empty:
    fig.add_trace(go.Scatter(x=pd.to_datetime(trades_plot["date"]), y=trades_plot["trades_price"], name="Trades-derived", line=dict(color="orange")))
fig.update_layout(title="Curve comparison", xaxis_title="Date", yaxis_title="Price", height=420)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# MTM & Insights
# ---------------------------
st.header("MTM & Insights")

trades_raw_for_ui = st.session_state.get("trades_raw", pd.DataFrame())
trade_label_map: Dict[str, str] = {}
trade_labels: List[str] = []
if isinstance(trades_raw_for_ui, pd.DataFrame) and not trades_raw_for_ui.empty:
    def _mk_label(r):
        tid = r.get("trade_id")
        spec = r.get("tradeSpecification_name") or ""
        period_label = r.get("period_defaultPeriodLabel") or r.get("period_relativePeriodLabel") or r.get("period_relativePeriodCode") or ""
        return f"{tid} | {spec} | {period_label}"

    trades_raw_for_ui = trades_raw_for_ui.reset_index(drop=True)
    labels = trades_raw_for_ui.apply(_mk_label, axis=1).tolist()
    ids = trades_raw_for_ui["trade_id"].astype(str).tolist()
    for lbl, tid in zip(labels, ids):
        trade_label_map[lbl] = tid
    trade_labels = labels

with st.expander("MTM settings"):
    st.write("Select valuation curve and trades (empty => all).")
    if trade_labels:
        selected_trade_labels = st.multiselect("Select trade(s) (empty => all)", options=trade_labels, default=[])
        selected_trade_ids = [trade_label_map[lbl] for lbl in selected_trade_labels]
    else:
        selected_trade_ids = []

    valuation_choice_mtm = st.selectbox("Valuation curve for MTM", options=["Assessments (period-average expanded)","Allocation (priority)","Trades-derived"], index=0)
    valuation_map_mtm = {
        "Assessments (period-average expanded)": st.session_state.get("assess_daily", pd.DataFrame()),
        "Allocation (priority)": st.session_state.get("alloc_curve", pd.DataFrame()),
        "Trades-derived": st.session_state.get("perperiod_daily", pd.DataFrame())
    }
    chosen_valuation_df_mtm = valuation_map_mtm.get(valuation_choice_mtm, st.session_state.get("assess_daily", pd.DataFrame()))
    mtm_multiplier_input = st.number_input("Unit multiplier / notional (per MWh)", value=float(st.session_state.get("mtm_multiplier", 1.0)), step=0.1)
    st.session_state["mtm_multiplier"] = float(mtm_multiplier_input)
    mtm_multiplier = float(mtm_multiplier_input)

    if st.button("Compute MTM & Insights", key="compute_mtm_button"):
        if chosen_valuation_df_mtm is None or chosen_valuation_df_mtm.empty:
            st.error("Chosen valuation curve is empty. Build curves or pick another valuation source.")
        else:
            trades_raw_df = st.session_state.get("trades_raw", pd.DataFrame())
            if not isinstance(trades_raw_df, pd.DataFrame) or trades_raw_df.empty:
                st.error("No trades loaded. Click 'Fetch trades' to load trades first.")
            else:
                with st.spinner("Expanding trades and computing MTM..."):
                    try:
                        trade_filter = selected_trade_ids if len(selected_trade_ids) > 0 else None
                        mtm_trades_daily_calc = compute_mtm_from_trades(trades_raw_df, chosen_valuation_df_mtm, trade_filter=trade_filter, multiplier=float(mtm_multiplier))
                        if mtm_trades_daily_calc is None or mtm_trades_daily_calc.empty:
                            st.warning("No expanded trade rows or no overlapping valuation dates — MTM produced an empty table.")
                            st.session_state["mtm_trades_daily"] = pd.DataFrame()
                            st.session_state["portfolio_daily"] = pd.DataFrame()
                        else:
                            st.session_state["mtm_trades_daily"] = mtm_trades_daily_calc
                            st.session_state["portfolio_daily"] = aggregate_portfolio_mtm(mtm_trades_daily_calc)
                            st.success("MTM computed and stored in session.")
                    except Exception as e:
                        logger.exception("MTM computation failed: %s", e)
                        st.error(f"MTM computation failed: {e}")

mtm_trades_daily = st.session_state.get("mtm_trades_daily", pd.DataFrame())
portfolio_daily = st.session_state.get("portfolio_daily", pd.DataFrame())

if mtm_trades_daily is None or mtm_trades_daily.empty:
    st.info("No MTM results yet. After fetching trades and building curves, compute MTM above.")
else:
    st.subheader("MTM: daily P&L and portfolio insights")
    st.markdown("Expanded trades (sample)")
    st.dataframe(mtm_trades_daily.head(200))

    if portfolio_daily is not None and not portfolio_daily.empty:
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Bar(x=pd.to_datetime(portfolio_daily["date"]), y=portfolio_daily["pnl_total"], name="Daily PnL", marker_color="indianred"))
        fig_pnl.add_trace(go.Scatter(x=pd.to_datetime(portfolio_daily["date"]), y=portfolio_daily["cum_pnl"], name="Cumulative PnL", line=dict(color="green")))
        fig_pnl.update_layout(title="Portfolio Daily and Cumulative PnL", xaxis_title="Date", yaxis_title="PNL", height=450)
        st.plotly_chart(fig_pnl, use_container_width=True)

        fig_hist = px.histogram(portfolio_daily, x="pnl_total", nbins=50, title="Daily PnL distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

        portfolio_daily = portfolio_daily.sort_values("date").reset_index(drop=True)
        try:
            pnl_series = pd.to_numeric(portfolio_daily.get("pnl_total", pd.Series(dtype="float")), errors="coerce")
            window = int(mtm_roll_days)
            if window <= 0:
                window = 20
            min_periods = min(5, window) if window > 0 else 1
            portfolio_daily["pnl_rolling_std"] = pnl_series.rolling(window=window, min_periods=min_periods).std()
        except Exception as e:
            logger.exception("Failed computing pnl_rolling_std: %s", e)
            portfolio_daily["pnl_rolling_std"] = np.nan

        try:
            portfolio_daily["rolling_var_parametric"] = portfolio_daily["pnl_rolling_std"].apply(lambda s: float(abs(s * mtm_z)) if not pd.isna(s) else np.nan)
        except Exception as e:
            logger.exception("Failed computing rolling_var_parametric: %s", e)
            portfolio_daily["rolling_var_parametric"] = np.nan

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=pd.to_datetime(portfolio_daily["date"]), y=portfolio_daily["pnl_rolling_std"], name=f"{mtm_roll_days}d rolling std", line=dict(color="blue")))
        fig_roll.add_trace(go.Scatter(x=pd.to_datetime(portfolio_daily["date"]), y=portfolio_daily["rolling_var_parametric"], name=f"{mtm_roll_days}d parametric VaR (z={mtm_z})", line=dict(color="red")))
        fig_roll.update_layout(title=f"Rolling {mtm_roll_days}-day volatility and parametric VaR", xaxis_title="Date", yaxis_title="Value", height=420)
        st.plotly_chart(fig_roll, use_container_width=True)

        total_pnl = float(portfolio_daily["pnl_total"].sum())
        cum_pnl_end = float(portfolio_daily["cum_pnl"].iloc[-1])
        best_day = portfolio_daily.loc[portfolio_daily["pnl_total"].idxmax()]
        worst_day = portfolio_daily.loc[portfolio_daily["pnl_total"].idxmin()]
        st.markdown("### Quick insights")
        st.write(f"- Total PnL over period: {total_pnl:.2f}")
        st.write(f"- Cumulative PnL (last day): {cum_pnl_end:.2f}")
        st.write(f"- Best day: {best_day['date']} → daily PnL = {best_day['pnl_total']:.2f}")
        st.write(f"- Worst day: {worst_day['date']} → daily PnL = {worst_day['pnl_total']:.2f}")

        pnl_series_full = portfolio_daily.get("pnl_total") if isinstance(portfolio_daily, pd.DataFrame) else None
        param_var = compute_parametric_var(pnl_series_full, 1.0, z=mtm_z)
        hist_var = compute_historical_var(pnl_series_full, 1.0, alpha=0.95)
        pv_str = f"{param_var:.2f}" if np.isfinite(param_var) else "n/a"
        hv_str = f"{hist_var:.2f}" if np.isfinite(hist_var) else "n/a"
        st.write(f"- Parametric VaR (z={mtm_z}) on daily pnl series (portfolio scale): {pv_str}")
        st.write(f"- Historical VaR (95%) on daily pnl series (portfolio scale): {hv_str}")

        contrib_period = top_contributors_by_period(mtm_trades_daily, top_n=15)
        if not contrib_period.empty:
            st.markdown("### Top contributors (by period)")
            st.dataframe(contrib_period)
            st_download_df("Download contributors by period CSV", contrib_period, file_name="mtm_top_contributors_by_period.csv", mime="text/csv")

            top_periods = contrib_period["period_key"].astype(str).tolist()[:10]
            if top_periods:
                stack_df = mtm_trades_daily.copy()
                def _row_period_key(r):
                    rel = r.get("period_relativePeriodLabel") or r.get("period_defaultPeriodLabel") or r.get("period_relativePeriodCode") or ""
                    abs_code = r.get("period_absolutePeriodCode") or ""
                    if rel and abs_code:
                        return f"{rel} | {abs_code}"
                    if rel:
                        return rel
                    if abs_code:
                        return abs_code
                    return str(r.get("trade_id"))

                if "period_key" not in stack_df.columns:
                    stack_df["period_key"] = stack_df.apply(_row_period_key, axis=1)

                stack_df = stack_df[stack_df["period_key"].astype(str).isin(top_periods)].copy()
                if stack_df.empty:
                    st.write("No rows available to plot stacked PnL for top periods.")
                else:
                    stack_agg = stack_df.groupby(["date","period_key"], dropna=False).agg(pnl=("pnl","sum")).reset_index()
                    stack_pivot = stack_agg.pivot(index="date", columns="period_key", values="pnl").fillna(0)
                    fig_stack = go.Figure()
                    for col in stack_pivot.columns:
                        fig_stack.add_trace(go.Bar(x=pd.to_datetime(stack_pivot.index), y=stack_pivot[col], name=str(col)))
                    fig_stack.update_layout(barmode="stack", title="Stacked daily PnL for top periods", xaxis_title="Date", yaxis_title="PNL", height=450)
                    st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("### Download MTM tables")
    st_download_df("Download expanded trades daily MTM CSV", mtm_trades_daily, file_name="mtm_trades_daily.csv", mime="text/csv")
    st_download_df("Download portfolio daily MTM CSV", portfolio_daily, file_name="mtm_portfolio_daily.csv", mime="text/csv")

# ============================================================
# IMPLIED VOLATILITY & VOLATILITY ANALYSIS SECTION
# ============================================================
st.header("Implied Volatility & Volatility Analysis")

st.markdown("""
This section computes volatility metrics from the trades data:
- Realized Volatility: Intra-period price dispersion from trades
- Volume-Weighted Volatility: Volatility normalized by trade size
- Bid-Ask Spread: Liquidity proxy
- IV Proxy: Combined metric (spread + moneyness + concentration)
- Rolling Curve Volatility: Day-over-day changes in the forward curve
- Term Structure Volatility: Spread volatility between periods
""")

with st.expander("IV Computation Settings & Triggers"):
    st.write("Configure and trigger volatility analysis.")
    iv_window = st.slider("Rolling window for curve vol (days)", min_value=5, max_value=60, value=20)

    if st.button("Compute IV Metrics", key="compute_iv_button"):
        trades_raw_df = st.session_state.get("trades_raw", pd.DataFrame())
        period_prices_assess = st.session_state.get("period_prices_assess", pd.DataFrame())
        assess_daily = st.session_state.get("assess_daily", pd.DataFrame())

        if trades_raw_df is None or trades_raw_df.empty:
            st.error("No trades data. Fetch trades first.")
        else:
            with st.spinner("Computing volatility metrics..."):
                try:
                    realized_vol_df = compute_price_volatility_by_period(trades_raw_df)
                    st.session_state["realized_vol_df"] = realized_vol_df

                    vw_vol_df = compute_volume_weighted_volatility(trades_raw_df)
                    st.session_state["vw_vol_df"] = vw_vol_df

                    iv_proxy_df = compute_moneyness_and_iv_proxy(trades_raw_df, period_prices_assess)
                    st.session_state["iv_proxy_df"] = iv_proxy_df

                    rolling_curve_vol = compute_rolling_curve_volatility(assess_daily, window=iv_window)
                    st.session_state["rolling_curve_vol"] = rolling_curve_vol

                    ts_vol_df = compute_term_structure_vol(period_prices_assess)
                    st.session_state["ts_vol_df"] = ts_vol_df

                    st.success("IV metrics computed and stored in session.")
                except Exception as e:
                    logger.exception("IV computation failed: %s", e)
                    st.error(f"IV computation failed: {e}")

realized_vol_df = st.session_state.get("realized_vol_df", pd.DataFrame())
vw_vol_df = st.session_state.get("vw_vol_df", pd.DataFrame())
iv_proxy_df = st.session_state.get("iv_proxy_df", pd.DataFrame())
rolling_curve_vol = st.session_state.get("rolling_curve_vol", pd.DataFrame())
ts_vol_df = st.session_state.get("ts_vol_df", pd.DataFrame())

if not realized_vol_df.empty:
    st.subheader("1. Realized Volatility by Period")
    st.markdown("""
    **Realized Volatility** measures how much individual trade prices within a period deviate from the mean.
    - Price Std: Standard deviation of trade prices
    - Price CV: Coefficient of variation (σ/μ) as % — normalized vol
    - Log-Return Vol: Volatility of log-returns between consecutive trades
    - Realized Vol: (Max - Min) / Mid — simple range-based estimate
    """)
    st.dataframe(realized_vol_df)
    st_download_df("Download realized vol CSV", realized_vol_df, file_name="realized_volatility.csv")

if not vw_vol_df.empty:
    st.subheader("2. Volume-Weighted Volatility by Period")
    st.markdown("""
    **Volume-Weighted Volatility** considers trade size when computing volatility.
    - VWAP: Volume-weighted average price
    - Volume-Weighted Vol: Standard deviation weighted by trade volume
    - Total Volume: Sum of all volumes in the period
    """)
    st.dataframe(vw_vol_df)
    st_download_df("Download VW vol CSV", vw_vol_df, file_name="volume_weighted_volatility.csv")

if not iv_proxy_df.empty:
    st.subheader("3. Implied Volatility Proxy & Moneyness by Period")
    st.markdown("""
    **IV Proxy** combines:
    - Bid-Ask Spread %: (Ask - Bid) / Mid
    - Moneyness: (Trade Price / Assessment Price - 1) × 100
    - Herfindahl Index: Trade concentration
    - IV Proxy %: Weighted combination reflecting inferred volatility
    """)
    tab1, tab2, tab3 = st.tabs(["IV Proxy Table", "IV Proxy Chart", "Moneyness Analysis"])

    with tab1:
        st.dataframe(iv_proxy_df)
        st_download_df("Download IV proxy CSV", iv_proxy_df, file_name="iv_proxy_moneyness.csv")

    with tab2:
        if not iv_proxy_df.empty:
            fig_iv = px.bar(
                iv_proxy_df.sort_values("iv_proxy_pct", ascending=False),
                x="period_label",
                y="iv_proxy_pct",
                color="spread_pct",
                title="Implied Volatility Proxy by Period",
                hover_data=["n_trades","bid_price","ask_price","spread_pct"]
            )
            fig_iv.update_layout(height=500, xaxis_title="Period", yaxis_title="IV Proxy (%)")
            st.plotly_chart(fig_iv, use_container_width=True)

    with tab3:
        if not iv_proxy_df[iv_proxy_df["moneyness_mean"].notna()].empty:
            fig_money = px.scatter(
                iv_proxy_df,
                x="moneyness_mean",
                y="moneyness_std",
                size="n_trades",
                color="iv_proxy_pct",
                hover_name="period_label",
                title="Moneyness Distribution by Period",
                labels={
                    "moneyness_mean":"Mean Moneyness (%)",
                    "moneyness_std":"Moneyness Std Dev (%)",
                    "iv_proxy_pct":"IV Proxy (%)"
                }
            )
            fig_money.update_layout(height=500)
            st.plotly_chart(fig_money, use_container_width=True)

if not rolling_curve_vol.empty:
    st.subheader("4. Rolling Curve Volatility (Log-Returns)")
    st.markdown(f"""
    **Rolling Curve Volatility** measures day-over-day changes in the forward curve.
    Window: {iv_window} days
    """)
    tab_curve1, tab_curve2 = st.tabs(["Chart", "Table"])
    with tab_curve1:
        fig_rolling = go.Figure()
        fig_rolling.add_trace(go.Scatter(
            x=pd.to_datetime(rolling_curve_vol["date"]),
            y=rolling_curve_vol["rolling_curve_vol"],
            name=f"{iv_window}d rolling vol",
            line=dict(color="purple", width=2),
            fill="tozeroy",
            fillcolor="rgba(128, 0, 128, 0.2)"
        ))
        fig_rolling.update_layout(
            title=f"Forward Curve {iv_window}-Day Rolling Volatility (log-returns %)",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig_rolling, use_container_width=True)

    with tab_curve2:
        st.dataframe(rolling_curve_vol)
        st_download_df("Download rolling curve vol CSV", rolling_curve_vol, file_name="rolling_curve_volatility.csv")

if not ts_vol_df.empty:
    st.subheader("5. Term Structure Volatility")
    st.markdown("""
    **Term Structure Volatility** measures the volatility of spreads between different periods.
    """)
    if not ts_vol_df.empty and ts_vol_df["ts_vol"].notna().any():
        fig_ts = px.area(
            ts_vol_df,
            x="date",
            y="ts_vol",
            title="Term Structure Volatility Over Time",
            labels={"ts_vol":"TS Volatility","date":"Date"}
        )
        fig_ts.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)

    st.dataframe(ts_vol_df)
    st_download_df("Download TS vol CSV", ts_vol_df, file_name="term_structure_volatility.csv")

# Summary statistics
st.subheader("Volatility Summary Statistics")
summary_stats = {}

if not realized_vol_df.empty and "realized_vol" in realized_vol_df.columns:
    rv = realized_vol_df["realized_vol"].dropna()
    if not rv.empty:
        summary_stats["Realized Vol (%)"] = {
            "Mean": f"{rv.mean():.4f}",
            "Median": f"{rv.median():.4f}",
            "Std": f"{rv.std():.4f}",
            "Min": f"{rv.min():.4f}",
            "Max": f"{rv.max():.4f}"
        }

if not iv_proxy_df.empty and "iv_proxy_pct" in iv_proxy_df.columns:
    iv = iv_proxy_df["iv_proxy_pct"].dropna()
    if not iv.empty:
        summary_stats["IV Proxy (%)"] = {
            "Mean": f"{iv.mean():.4f}",
            "Median": f"{iv.median():.4f}",
            "Std": f"{iv.std():.4f}",
            "Min": f"{iv.min():.4f}",
            "Max": f"{iv.max():.4f}"
        }

if not rolling_curve_vol.empty and "rolling_curve_vol" in rolling_curve_vol.columns:
    rcv = rolling_curve_vol["rolling_curve_vol"].dropna()
    if not rcv.empty:
        summary_stats["Rolling Curve Vol (%)"] = {
            "Mean": f"{rcv.mean():.4f}",
            "Median": f"{rcv.median():.4f}",
            "Std": f"{rcv.std():.4f}",
            "Min": f"{rcv.min():.4f}",
            "Max": f"{rcv.max():.4f}"
        }

if summary_stats:
    summary_df = pd.DataFrame(summary_stats).T
    st.dataframe(summary_df)
    st_download_df("Download summary stats CSV", summary_df.reset_index().rename(columns={"index":"Metric"}), file_name="volatility_summary_stats.csv")
else:
    st.info("No volatility metrics computed yet. Run 'Compute IV Metrics' to populate this section.")

st.markdown("---")
st.markdown(
    """
    ### How to use this app:
    1. Build a PriceAssessment curve: select a specification and click the button in the sidebar
    2. Fetch trades: select trade specifications, then click 'Fetch trades'
    3. Compute MTM: select a valuation curve and click 'Compute MTM & Insights'
    4. Analyze IV: click 'Compute IV Metrics' to generate volatility analysis
    5. Build location spreads: select multiple hub specs, choose a reference hub, and click 'Build location spreads'
    
    Only matching delivery periods are used in the spread calculation, so 6-month and 4-month curves
    only generate a spread for the 4 overlapping months.
    """
)