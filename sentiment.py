# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 10:19:06 2025

@author: tson
"""

#!/usr/bin/env python3
"""
VN Stock Sentiment from 100+ Vietnam News RSS
--------------------------------------------
- Thu thập tin tức qua RSS từ >100 nguồn (tổng hợp, tài chính, địa phương).
- Trích văn bản bài viết (trafilatura), lọc bài tiếng Việt & liên quan chứng khoán.
- Phân tích sentiment đa ngôn ngữ bằng HuggingFace (xlm-roberta-base-sentiment).
- Xuất kết quả chi tiết (CSV/JSONL) + tổng hợp theo mã cổ phiếu theo ngày.

Cài đặt nhanh
-------------
python -m pip install --upgrade pip
python -m pip install feedparser trafilatura transformers torch pandas numpy langdetect python-dateutil rapidfuzz tqdm

Chạy ví dụ
----------
python vn_stock_sentiment_rss.py --days-back 3 --max-per-feed 40 --outdir out \
  --tickers-file tickers.txt --feeds-file extra_feeds.txt

Ghi chú
-------
- Tôn trọng robots.txt/điều khoản website. Ưu tiên RSS.
- Bạn có thể mở rộng/giảm danh sách FEEDS hoặc truyền file ngoài.
- Mặc định chỉ nhận diện tickers bằng danh sách KNOWN_TICKERS + tickers.txt.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import feedparser
import pandas as pd
from dateutil import parser as dateparser
from langdetect import detect as lang_detect, LangDetectException
from rapidfuzz import fuzz
from tqdm import tqdm

# Optional imports with graceful fallback
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except Exception as e:
    print("[WARN] trafilatura not available:", e)
    TRAFILATURA_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print("[WARN] transformers not available:", e)
    TRANSFORMERS_AVAILABLE = False

# ------------------
# >100 curated RSS feeds (general + business + regional)
# ------------------
FEEDS = [
    # National general news
    "https://vnexpress.net/rss/tin-moi-nhat.rss",
    "https://vnexpress.net/rss/kinh-doanh.rss",
    "https://tuoitre.vn/rss.htm",
    "https://thanhnien.vn/rss/home.rss",
    "https://vietnamnet.vn/rss/home.rss",
    "https://zingnews.vn/rss.html",
    "https://laodong.vn/rss/home.rss",
    "https://dantri.com.vn/rss/home.rss",
    "https://nhandan.vn/rss/home.rss",
    "https://baomoi.com/rss/general.rss",
    "https://vtv.vn/kinh-te.rss",
    "https://vov.vn/rss",

    # Business/finance core
    "https://cafef.vn/trang-chu.rss",
    "https://cafebiz.vn/rss.chn",
    "https://vietstock.vn/rss/home.rss",
    "https://s.cafef.vn/rss/",
    "https://ndh.vn/rss/home.rss",
    "https://vneconomy.vn/feed",
    "https://vir.com.vn/rss/",
    "https://baodautu.vn/rss/",
    "https://www.tinnhanhchungkhoan.vn/rss/",
    "https://nhipcaudautu.vn/rss/",
    "https://tapchitaichinh.vn/rss/tin-tuc.rss",
    "https://thoibaonganhang.vn/rss/home.rss",
    "https://thoibaotaichinhvietnam.vn/rss/home.rss",
    "https://haiquanonline.com.vn/rss/home.rss",
    "https://bnews.vn/rss",
    "https://www.vietnambiz.vn/kinh-doanh.rss",
    "https://congthuong.vn/rss/home.rss",

    # Securities companies / research (where RSS offered)
    "https://www.hsc.com.vn/feed/",
    "https://miraeasset.com.vn/feed/",
    "https://kisvn.vn/feed/",
    "https://www.mbs.com.vn/feed/",
    "https://www.ssi.com.vn/rss",
    "https://yuanta.com.vn/feed/",
    "https://www.vcbs.com.vn/News/Feed",
    "https://www.bsc.com.vn/Feed",

    # English-language Vietnam business
    "https://e.vnexpress.net/rss/business.rss",
    "https://vietnamnews.vn/rss/economy.rss",
    "https://tuoitrenews.vn/rss",
    "https://vir.com.vn/rss/",
    "https://saigontimes.com.vn/en/feed/",

    # Specialized / magazines
    "https://theleader.vn/rss/trang-chu.rss",
    "https://doanhnhan.vn/rss",
    "https://doanhnhansaigon.vn/rss",
    "https://thuongtruong.com.vn/rss",
    "https://thuonghieucongluan.com.vn/rss/home.rss",
    "https://saigontimes.vn/feed/",
    "https://vietnamfinance.vn/rss.htm",
    "https://forbesvietnam.com.vn/feed/",

    # Regional newspapers (subset >60)
    "https://baodongnai.com.vn/rss/home.rss",
    "https://baobinhduong.vn/rss/home.rss",
    "https://baobariavungtau.com.vn/rss/home.rss",
    "https://baobinhthuan.com.vn/rss/home.rss",
    "https://baokhanhhoa.vn/rss/home.rss",
    "https://baonghean.vn/rss/home.rss",
    "https://baodaklak.vn/rss/home.rss",
    "https://baodaknong.org.vn/rss/home.rss",
    "https://baolamdong.vn/rss/home.rss",
    "https://baonamdinh.vn/rss/home.rss",
    "https://baothaibinh.com.vn/rss/home.rss",
    "https://baothanhhoa.vn/rss/home.rss",
    "https://baothainguyen.vn/rss/home.rss",
    "https://baophuquoc.vn/rss/home.rss",
    "https://baocantho.com.vn/rss/home.rss",
    "https://baolongan.vn/rss/home.rss",
    "https://baodongthap.vn/rss/home.rss",
    "https://baoangiang.com.vn/rss/home.rss",
    "https://baotravinh.vn/rss/home.rss",
    "https://baovinhlong.vn/rss/home.rss",
    "https://baotayninh.vn/rss/home.rss",
    "https://baoquangninh.vn/rss/home.rss",
    "https://baohaiphong.com.vn/rss/home.rss",
    "https://baobacgiang.vn/rss/home.rss",
    "https://baobaclieu.vn/rss/home.rss",
    "https://baohatinh.vn/rss/home.rss",
    "https://baophutho.vn/rss/home.rss",
    "https://baogialai.com.vn/rss/home.rss",
    "https://baokontum.com.vn/rss/home.rss",
    "https://baolaichau.vn/rss/home.rss",
    "https://baolaocai.vn/rss/home.rss",
    "https://baosonla.org.vn/rss/home.rss",
    "https://baodienbienphu.com.vn/rss/home.rss",
    "https://baoyenbai.com.vn/rss/home.rss",
    "https://baotuyenquang.com.vn/rss/home.rss",
    "https://baobackan.com.vn/rss/home.rss",
    "https://baocaobang.vn/rss/home.rss",
    "https://baobinhdinh.vn/rss/home.rss",
    "https://baoquangnam.vn/rss/home.rss",
    "https://baoquangngai.vn/rss/home.rss",
    "https://baophuyen.vn/rss/home.rss",
    "https://baokhanhhoa.vn/rss/kinh-te.rss",
    "https://baonghean.vn/rss/kinh-te.rss",
    "https://baothanhhoa.vn/rss/kinh-te.rss",
    "https://baohaiduong.vn/rss/home.rss",
    "https://baoninhbinh.vn/rss/home.rss",
    "https://baohungyen.vn/rss/home.rss",
    "https://baobariavungtau.com.vn/rss/kinh-te.rss",
    "https://baobinhduong.vn/rss/kinh-te.rss",
    "https://baocaobang.vn/rss/kinh-te.rss",
    "https://baobinhthuan.com.vn/rss/kinh-te.rss",
    "https://baobackan.com.vn/rss/kinh-te.rss",
    "https://baodongnai.com.vn/rss/kinh-te.rss",
    "https://baobinhphuoc.com.vn/rss/home.rss",
    "https://baobacninh.com.vn/rss/home.rss",
    "https://baonguoiduatin.vn/rss/home.rss",

    # International wires referencing Vietnam (optional)
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.bloomberg.com/feeds/podcasts.xml",
    "https://www.ft.com/?format=rss",
]

# ------------------
# Known VN stock tickers (seed subset) + extend via --tickers-file
# ------------------
KNOWN_TICKERS = {
    "VIC","VHM","VNM","VCB","BID","CTG","TCB","VPB","MBB","ACB",
    "GAS","HPG","FPT","MWG","SSI","GVR","VJC","PLX","STB","POW",
    "PNJ","BVH","VRE","SAB","REE","HDB","NVL","KDH","DXG","PDR",
    "VHC","DPM","DGC","VTP","VIB","SHB","LPB","TPB","NKG","DHA",
}

FINANCE_KEYWORDS = [
    "chứng khoán","cổ phiếu","VN-Index","VNIndex","HNX-Index","UPCoM",
    "thị trường","nhà đầu tư","niêm yết","phát hành","trái phiếu",
    "lợi nhuận","doanh thu","kết quả kinh doanh","chia cổ tức","M&A",
]

# ------------------
# Helpers
# ------------------

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return dateparser.parse(dt_str)
    except Exception:
        return None


def is_vietnamese(text: str) -> bool:
    try:
        lang = lang_detect(text)
        return lang == "vi"
    except LangDetectException:
        return True

TICKER_RE = re.compile(r"\b[A-Z]{3,5}\b")


def find_tickers(text: str) -> List[str]:
    candidates = set(TICKER_RE.findall(text.upper()))
    return sorted(list(candidates & KNOWN_TICKERS))


def contains_finance_keywords(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in FINANCE_KEYWORDS)

# ------------------
# Content extraction
# ------------------

def extract_text(url: str, timeout: int = 15) -> str:
    if TRAFILATURA_AVAILABLE:
        trafilatura.settings.use_config("DEFAULT")
        downloaded = trafilatura.fetch_url(
            url,
            timeout=timeout,
            no_ssl=False,
            user_agent="Mozilla/5.0 (compatible; VNStockSentiment/1.0)"
        )
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_images=False,
                include_tables=False,
                no_fallback=False,
            )
            if text:
                return text
    return ""

# ------------------
# Sentiment pipeline
# ------------------

class Sentiment:
    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is not installed. Please `pip install transformers torch`.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            truncation=True,
        )

    def score(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        text = text.strip()
        if not text:
            return ("neutral", 0.0, {"negative": 0.0, "neutral": 1.0, "positive": 0.0})
        res = self.pipe(text[:2000])  # process first 2000 chars
        scores = {d["label"].lower(): float(d["score"]) for d in res[0]}
        mapping = {
            "positive": "positive",
            "neutral": "neutral",
            "negative": "negative",
            "LABEL_2": "positive",
            "LABEL_1": "neutral",
            "LABEL_0": "negative",
        }
        normalized = {mapping.get(k, k): v for k, v in scores.items()}
        label = max(normalized, key=normalized.get)
        conf = normalized[label]
        return (label, float(conf), normalized)

# ------------------
# Main pipeline
# ------------------

def fetch_from_feed(feed_url: str, max_items: int, days_back: int) -> List[dict]:
    out = []
    fp = feedparser.parse(feed_url)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    for entry in fp.entries[: max_items or None]:
        link = entry.get("link")
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        published = parse_dt(entry.get("published") or entry.get("updated"))
        if published and published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published and published < cutoff:
            continue
        out.append({
            "source": feed_url,
            "title": title,
            "summary": summary,
            "link": link,
            "published": published.isoformat() if published else None,
        })
    return out


def dedupe_entries(entries: List[dict]) -> List[dict]:
    seen = set()
    deduped: List[dict] = []
    for e in entries:
        url = e.get("link")
        title = (e.get("title") or "").strip()
        key = url or title
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    final: List[dict] = []
    for e in deduped:
        if any(fuzz.token_set_ratio(e.get("title",""), x.get("title","")) > 92 for x in final):
            continue
        final.append(e)
    return final


def process(entries: List[dict], senti: Sentiment, rate_delay: float = 0.3) -> List[dict]:
    rows = []
    for e in tqdm(entries, desc="Scoring"):
        url = e.get("link") or ""
        title = e.get("title") or ""
        summary = e.get("summary") or ""
        text = extract_text(url)
        candidate_text = text or (title + ". " + summary)
        if not candidate_text.strip():
            continue
        if not is_vietnamese(candidate_text):
            continue
        tickers = find_tickers(title + " " + candidate_text)
        if not tickers and not contains_finance_keywords(candidate_text + " " + title):
            continue
        label, conf, scores = senti.score(title + ". " + candidate_text[:2000])
        rows.append({
            "timestamp_utc": utcnow_iso(),
            "published": e.get("published"),
            "source": e.get("source"),
            "url": url,
            "title": title,
            "sentiment": label,
            "confidence": round(conf, 4),
            "score_positive": round(scores.get("positive", 0.0), 4),
            "score_neutral": round(scores.get("neutral", 0.0), 4),
            "score_negative": round(scores.get("negative", 0.0), 4),
            "tickers": ",".join(tickers),
            "has_finance_kw": contains_finance_keywords(candidate_text),
        })
        time.sleep(rate_delay)
    return rows


def aggregate_by_ticker(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    def to_date(x):
        try:
            dt = dateparser.parse(x)
            return dt.date().isoformat() if dt else None
        except Exception:
            return None
    df["date"] = df["published"].apply(to_date)
    df["tickers_list"] = df["tickers"].apply(lambda s: [t for t in s.split(",") if t] if isinstance(s, str) else [])
    df = df.explode("tickers_list")
    df = df[df["tickers_list"].notna() & (df["tickers_list"] != "")]
    if df.empty:
        return pd.DataFrame()
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_num"] = df["sentiment"].map(mapping)
    df["polarity"] = df["score_positive"] - df["score_negative"]
    grp = df.groupby(["tickers_list", "date"], as_index=False).agg(
        n_articles=("url", "count"),
        avg_sentiment_num=("sentiment_num", "mean"),
        avg_polarity=("polarity", "mean"),
        avg_confidence=("confidence", "mean"),
    ).rename(columns={"tickers_list": "ticker"})
    return grp


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_outputs(rows: List[dict], outdir: str):
    ensure_dir(outdir)
    articles_csv = os.path.join(outdir, "articles.csv")
    articles_jsonl = os.path.join(outdir, "articles.jsonl")
    with open(articles_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["note"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(articles_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg_df = aggregate_by_ticker(rows)
    agg_path = os.path.join(outdir, "per_ticker_sentiment.csv")
    if not agg_df.empty:
        agg_df.to_csv(agg_path, index=False)
    print(f"Saved: {articles_csv}\n       {articles_jsonl}")
    if not agg_df.empty:
        print(f"       {agg_path}")
    else:
        print("No ticker-level aggregates (no tickers found).")

# ------------------
# CLI
# ------------------

def main():
    parser = argparse.ArgumentParser(description="Vietnam Stock Sentiment from News RSS")
    parser.add_argument("--days-back", type=int, default=3, help="Only include items published within the last N days")
    parser.add_argument("--max-per-feed", type=int, default=40, help="Max items per feed to fetch")
    parser.add_argument("--outdir", type=str, default="out", help="Output directory")
    parser.add_argument("--rate-delay", type=float, default=0.3, help="Delay (s) between article fetches to be polite")
    parser.add_argument("--min-sources", type=int, default=1, help="Minimum distinct RSS feeds required (sanity check)")
    parser.add_argument("--feeds-file", type=str, default=None, help="Optional path to a text file with extra RSS feed URLs (one per line)")
    parser.add_argument("--tickers-file", type=str, default=None, help="Optional path to a text file containing tickers (one per line)")
    parser.add_argument("--model", type=str, default="cardiffnlp/twitter-xlm-roberta-base-sentiment", help="HF model for sentiment")
    args = parser.parse_args()

    feeds = list(FEEDS)
    if args.feeds_file and os.path.isfile(args.feeds_file):
        with open(args.feeds_file, "r", encoding="utf-8") as f:
            extra = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            feeds.extend(extra)
    feeds = list(dict.fromkeys(feeds))  # unique preserve order

    # Extend tickers from file if provided
    if args.tickers_file and os.path.isfile(args.tickers_file):
        with open(args.tickers_file, "r", encoding="utf-8") as f:
            extra_tickers = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]
            KNOWN_TICKERS.update(set(extra_tickers))

    print(f"[INFO] Using {len(feeds)} RSS feeds")
    if len(feeds) < args.min_sources:
        print("[ERROR] Not enough sources given.")
        sys.exit(2)

    # Collect entries
    all_entries: List[dict] = []
    for url in tqdm(feeds, desc="Fetching RSS"):
        try:
            items = fetch_from_feed(url, args.max_per_feed, args.days_back)
            all_entries.extend(items)
        except Exception as e:
            print(f"[WARN] Failed feed {url}: {e}")
        time.sleep(0.1)

    print(f"[INFO] Collected {len(all_entries)} items before dedupe")
    entries = dedupe_entries(all_entries)
    print(f"[INFO] {len(entries)} items after dedupe")

    # Init sentiment model
    senti = Sentiment(model_name=args.model)

    # Process entries
    rows = process(entries, senti, rate_delay=args.rate_delay)

    # Save
    save_outputs(rows, args.outdir)


if __name__ == "__main__":
    main()
