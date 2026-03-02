#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese stock sentiment from multiple websites via RSS.

Outputs:
- out/articles_sentiment.csv
- out/summary_by_ticker.csv
- out/source_stats.csv
"""

import argparse
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import feedparser
import pandas as pd
import trafilatura
from langdetect import LangDetectException, detect
from transformers import pipeline

# Multi-website RSS list (general + business + finance)
FEEDS = [
    "https://vnexpress.net/rss/kinh-doanh.rss",
    "https://vnexpress.net/rss/tin-moi-nhat.rss",
    "https://tuoitre.vn/rss/kinh-doanh.rss",
    "https://tuoitre.vn/rss.htm",
    "https://thanhnien.vn/rss/kinh-doanh.rss",
    "https://thanhnien.vn/rss/home.rss",
    "https://vietnamnet.vn/rss/kinh-doanh.rss",
    "https://vietnamnet.vn/rss/home.rss",
    "https://laodong.vn/rss/home.rss",
    "https://dantri.com.vn/rss/home.rss",
    "https://cafef.vn/trang-chu.rss",
    "https://cafebiz.vn/rss.chn",
    "https://vietstock.vn/rss/home.rss",
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
    "https://congthuong.vn/rss/home.rss",
    "https://www.vietnambiz.vn/kinh-doanh.rss",
    "https://bnews.vn/rss",
    "https://nld.com.vn/kinh-te.rss",
    "https://vtv.vn/kinh-te.rss",
    "https://vov.vn/rss",
    "https://baomoi.com/rss/general.rss",
    "https://theleader.vn/rss/trang-chu.rss",
    "https://saigontimes.vn/feed/",
    "https://vietnamfinance.vn/rss.htm",
    "https://forbesvietnam.com.vn/feed/",
    "https://www.hsc.com.vn/feed/",
    "https://www.mbs.com.vn/feed/",
    "https://www.ssi.com.vn/rss",
    "https://yuanta.com.vn/feed/",
    "https://www.vcbs.com.vn/News/Feed",
    "https://www.bsc.com.vn/Feed",
]

VI_STOCK_KEYWORDS = [
    "chung khoan",
    "co phieu",
    "vn-index",
    "vnindex",
    "hnx-index",
    "upcom",
    "thi truong",
    "nha dau tu",
    "niem yet",
    "loi nhuan",
    "doanh thu",
    "ket qua kinh doanh",
    "co tuc",
    "trai phieu",
]

POSITIVE_WORDS = {
    "tang", "tang truong", "but pha", "dot bien", "lac quan", "vuot dinh", "mua rong",
    "ke hoach cao", "hoan thanh", "ky luc", "tot", "tich cuc", "hoi phuc", "nang hang",
    "mo rong", "gia tang", "ky vong", "kha quan", "dong tien vao", "kiem soat tot",
}

NEGATIVE_WORDS = {
    "giam", "lao doc", "sut giam", "thua lo", "lo", "bi phat", "dieu tra", "canh bao",
    "huy niem yet", "cat lo", "ban rong", "rui ro", "xau", "tieu cuc", "ap luc ban",
    "dong bang", "suy yeu", "giam sau", "bi ban thao", "mat thanh khoan", "pha day",
}

VI_TICKERS = {
    "ACB", "ANV", "AST", "BID", "BMI", "BMP", "BSI", "BVH", "BWE", "CMG",
    "CTD", "CTG", "CTR", "DBC", "DCM", "DGC", "DGW", "DHG", "DPM", "DRC",
    "DXG", "FCN", "FPT", "FRT", "FTS", "GAS", "GEX", "GMD", "GVR", "HAG",
    "HAH", "HCM", "HDB", "HDG", "HPG", "HSG", "KBC", "KDH", "LPB", "MBB",
    "MSN", "MWG", "NKG", "NLG", "OCB", "PAN", "PC1", "PDR", "PET", "PLX",
    "PNJ", "POW", "PVD", "PVT", "REE", "SAB", "SHB", "SSI", "STB", "TCB",
    "TPB", "VCB", "VCG", "VCI", "VGC", "VHC", "VHM", "VIB", "VIC", "VJC",
    "VND", "VNM", "VPB", "VRE",
}

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

TICKER_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{3,5}(?![A-Z0-9])")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?\n])\s+")


def strip_vietnamese_marks(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def normalize_text(text: str) -> str:
    lowered = text.lower()
    no_marks = strip_vietnamese_marks(lowered)
    return re.sub(r"\s+", " ", no_marks).strip()


def is_vi_stock_article(text: str) -> bool:
    normalized = normalize_text(text)
    return any(keyword in normalized for keyword in VI_STOCK_KEYWORDS)


def lexicon_sentiment_score(text: str) -> float:
    normalized = normalize_text(text)
    pos = sum(1 for w in POSITIVE_WORDS if w in normalized)
    neg = sum(1 for w in NEGATIVE_WORDS if w in normalized)
    total = pos + neg
    if total == 0:
        return 0.0
    # Bounded to [-1, 1]
    return (pos - neg) / total


def parse_entry_time(entry) -> datetime:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def detect_tickers(text: str, allowed: Set[str]) -> List[str]:
    found = set(TICKER_RE.findall(text.upper()))
    return sorted(found & allowed)


def split_sentences(text: str) -> List[str]:
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def pick_ticker_context(content: str, ticker: str, max_sentences: int = 6) -> str:
    sentences = split_sentences(content)
    if not sentences:
        return content[:1200]

    ticker_upper = ticker.upper()
    keyword_hits = []
    normalized_keywords = {normalize_text(k) for k in VI_STOCK_KEYWORDS}

    for sentence in sentences:
        up = sentence.upper()
        if ticker_upper not in up:
            continue
        score = 1
        normalized_sentence = normalize_text(sentence)
        if any(k in normalized_sentence for k in normalized_keywords):
            score += 1
        if any(w in normalized_sentence for w in POSITIVE_WORDS):
            score += 1
        if any(w in normalized_sentence for w in NEGATIVE_WORDS):
            score += 1
        keyword_hits.append((score, sentence))

    if not keyword_hits:
        return content[:1200]

    keyword_hits.sort(key=lambda x: x[0], reverse=True)
    selected = [sent for _, sent in keyword_hits[:max_sentences]]
    return "\n".join(selected)[:1200]


def fetch_article_text(url: str) -> str:
    raw = trafilatura.fetch_url(url)
    if not raw:
        return ""
    extracted = trafilatura.extract(raw, include_comments=False, include_tables=False)
    return extracted or ""


def crawl_feeds(feeds: List[str], days_back: int, max_per_feed: int) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    rows: List[Dict[str, str]] = []
    seen_links: Set[str] = set()

    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries[:max_per_feed]:
                link = entry.get("link", "").strip()
                if not link or link in seen_links:
                    continue
                published = parse_entry_time(entry)
                if published < cutoff:
                    continue
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                text = fetch_article_text(link)
                if not text:
                    text = f"{title}\n{summary}"
                rows.append(
                    {
                        "title": title,
                        "summary": summary,
                        "text": text,
                        "link": link,
                        "published": published.isoformat(),
                        "source_feed": feed_url,
                    }
                )
                seen_links.add(link)
        except Exception as exc:
            print(f"[WARN] cannot parse feed {feed_url}: {exc}")

    return pd.DataFrame(rows)


def build_sentiment_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    )


def detect_language(text: str) -> Optional[str]:
    try:
        return detect(text)
    except LangDetectException:
        return None


def extract_score_map(prediction_output) -> Dict[str, float]:
    # Handles transformers outputs across versions:
    # - [{"label":"LABEL_0","score":...}, ...]
    # - [[{"label":"LABEL_0","score":...}, ...]]
    if isinstance(prediction_output, list) and prediction_output:
        first = prediction_output[0]
        if isinstance(first, list):
            prediction_output = first
    if not isinstance(prediction_output, list):
        return {}
    return {item.get("label", ""): float(item.get("score", 0.0)) for item in prediction_output if isinstance(item, dict)}


def analyze_articles(
    df: pd.DataFrame,
    sentiment_model,
    allowed_tickers: Set[str],
    pos_threshold: float,
    neg_threshold: float,
    lexicon_weight: float,
) -> pd.DataFrame:
    results: List[Dict[str, str]] = []
    for row in df.itertuples(index=False):
        content = f"{row.title}\n{row.summary}\n{row.text}"
        if not is_vi_stock_article(content):
            continue

        lang = detect_language(content[:1000])
        if lang not in (None, "vi"):
            continue

        tickers = detect_tickers(content, allowed_tickers)
        if not tickers:
            continue

        for ticker in tickers:
            context_text = pick_ticker_context(content, ticker)
            scored = sentiment_model(context_text[:512], top_k=None)
            score_map = extract_score_map(scored)

            neg = score_map.get("LABEL_0", 0.0)
            neu = score_map.get("LABEL_1", 0.0)
            pos = score_map.get("LABEL_2", 0.0)

            model_score = pos - neg
            lex_score = lexicon_sentiment_score(context_text)
            model_weight = max(0.0, 1.0 - lexicon_weight)
            final_score = model_weight * model_score + lexicon_weight * lex_score

            if final_score >= pos_threshold:
                label = "positive"
            elif final_score <= neg_threshold:
                label = "negative"
            else:
                label = "neutral"

            results.append(
                {
                    "ticker": ticker,
                    "label": label,
                    "score": final_score,
                    "model_score": model_score,
                    "lexicon_score": lex_score,
                    "p_pos": pos,
                    "p_neu": neu,
                    "p_neg": neg,
                    "title": row.title,
                    "published": row.published,
                    "link": row.link,
                    "source_feed": row.source_feed,
                    "context_text": context_text,
                }
            )

    return pd.DataFrame(results)


def summarize_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "positive", "neutral", "negative", "articles", "avg_score", "net_mentions"]
        )

    grouped = df.groupby(["ticker", "label"]).size().unstack(fill_value=0)
    for label in ("positive", "neutral", "negative"):
        if label not in grouped.columns:
            grouped[label] = 0
    grouped["articles"] = grouped["positive"] + grouped["neutral"] + grouped["negative"]
    score_mean = df.groupby("ticker")["score"].mean().rename("avg_score")
    result = grouped.join(score_mean).reset_index()
    result["net_mentions"] = result["positive"] - result["negative"]
    return result.sort_values(["articles", "avg_score"], ascending=[False, False])


def summarize_by_source(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source_feed", "articles"])
    return df.groupby("source_feed").size().rename("articles").sort_values(ascending=False).reset_index()


def save_outputs(raw_df: pd.DataFrame, summary_df: pd.DataFrame, source_df: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    raw_df.to_csv(os.path.join(outdir, "articles_sentiment.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(outdir, "summary_by_ticker.csv"), index=False, encoding="utf-8-sig")
    source_df.to_csv(os.path.join(outdir, "source_stats.csv"), index=False, encoding="utf-8-sig")


def parse_args():
    parser = argparse.ArgumentParser(description="Vietnamese stock sentiment analysis from many RSS websites.")
    parser.add_argument("--days-back", type=int, default=3, help="Only keep articles from the last N days.")
    parser.add_argument("--max-per-feed", type=int, default=50, help="Maximum entries per feed.")
    parser.add_argument("--outdir", type=str, default="out", help="Output folder.")
    parser.add_argument("--tickers-file", type=str, default="", help="Optional text file with one ticker per line.")
    parser.add_argument("--pos-threshold", type=float, default=0.08, help="Min score to classify positive.")
    parser.add_argument("--neg-threshold", type=float, default=-0.08, help="Max score to classify negative.")
    parser.add_argument("--lexicon-weight", type=float, default=0.25, help="Weight of lexicon score in final score [0..1].")
    return parser.parse_args()


def load_tickers(extra_file: str) -> Set[str]:
    tickers = set(VI_TICKERS)
    if extra_file and os.path.exists(extra_file):
        with open(extra_file, "r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip().upper()
                if re.fullmatch(r"[A-Z]{3,5}", value):
                    tickers.add(value)
    return tickers


def main():
    args = parse_args()
    tickers = load_tickers(args.tickers_file)

    print("[1/4] Crawling RSS feeds...")
    crawled = crawl_feeds(FEEDS, args.days_back, args.max_per_feed)
    print(f"Collected {len(crawled)} articles.")

    print("[2/4] Loading sentiment model...")
    sentiment_model = build_sentiment_pipeline()

    print("[3/4] Running sentiment analysis...")
    lexicon_weight = min(max(args.lexicon_weight, 0.0), 1.0)
    raw_results = analyze_articles(
        crawled,
        sentiment_model,
        tickers,
        args.pos_threshold,
        args.neg_threshold,
        lexicon_weight,
    )
    print(f"Scored {len(raw_results)} ticker-level sentiment rows.")

    print("[4/4] Saving outputs...")
    summary_results = summarize_by_ticker(raw_results)
    source_results = summarize_by_source(raw_results)
    save_outputs(raw_results, summary_results, source_results, args.outdir)
    print(f"Done. Files are in: {args.outdir}")


if __name__ == "__main__":
    main()
