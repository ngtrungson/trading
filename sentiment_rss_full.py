import os
import argparse
import feedparser
import trafilatura
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from datetime import datetime, timedelta, timezone
from transformers import pipeline
from langdetect import detect
from tqdm import tqdm

# ------------------------------
# Danh sách 100 mã HSX (cập nhật từ HOSE)
# ------------------------------
HSX_TICKERS = [
    "ACB","ANV","BCM","BID","BMP","BVH","CMG","CNG","CTD","CTG",
    "DHG","DIG","DPM","DRC","DXG","EIB","FPT","GAS","GMD","GVR",
    "HAG","HAH","HBC","HCM","HPG","HSG","HT1","HVN","IDI","IMP",
    "KBC","KDC","KDH","LPB","MBB","MSN","MWG","NKG","NLG","NVB",
    "OCB","PAC","PAN","PET","PGD","PHR","PLX","PNJ","POW","PVD",
    "PVT","REE","ROS","SAB","SAM","SAV","SBT","SCR","SHB","SHI",
    "SSI","STB","SZC","TCB","TDH","TPB","VCB","VCI","VGC","VHM",
    "VIB","VIC","VID","VJC","VND","VNG","VNL","VNM","VPB","VPI",
    "VPS","VRC","VRE","VSH","VSI","YEG","HAP","NT2","QCG","TCH",
    "GEX","FCN","VHG","HTN","DRH","ASM","AAA","DBC","NVL","HPX"
]


HSX_TICKERS = ['IDC', 'IDV', 'NTP', 'PVS',  'PLC', 'SHS', 'TNG',  'VCS', 'CDN','VNR',
               'ANV',  "ACB", 'AST','ABT',
              "BWE",  "BID", "BMI", "BMP", "BVH", 'BFC', 'BCM', 'BSI', 'BIC',
              'CMG', "CTD", "CSV", "CTG", 'CII', 'CTS', 'CTR', 'CTI',
              'D2D', 'DGW', 'DBC', "DHG",  "DPM",  "DRC", "DVP", 'DHA', 'DCM', 'DSE', 'DGC', 'DHC',
              'FRT', "FCN",  'FMC', "FPT", 'FTS',
              "GAS", "GMD", 'GVR', 'GIL', 'GEX','GEE',
              "HSG",  'HHV', "HDG", "HCM", "HPG",  'HDC', 'HAH', "HDB", 'HTI',
              'IMP', "IJC", 'ILB',  'ITD',
              "KBC",  "KDH", 'KSB',
              'LHG', 'LCG', "LPB",
              "MBB", "MSN", "MWG",  'MSH', 'MBS',
              "NLG", 'NTL', "NKG", 'NCT', 'OCB',
              "PVT", "PVD", "PHR", "PNJ",  "PC1",   "PLX", "PPC", 'PTB', 'PVP', 'POW', 'PET','PVP','PGV',
              "REE", "SJS", "STB", "SSI", "SBT",  'SKG', 'SZL', 'SZC', 'SHB', 'SGN',
              "TIP", "TCL", 'TDM', 'TCM',  'TCB', 'TNH', 'TYA',
              "VNM", "VHC", "VIC", "VCB", "VSC", "VJC", "VIB", 'VGC', 'VPB', 'VRE', 'VND','VCP',
              'VHM',  'VCI', 'VTP', 'VCG',
              'QNS',  'ACV', 'VGI', 'PPH', 'DRI','VLB','PAP', 'PDV','NTC',
                'PHP', 'VEA', 'VGT', 'SNZ', 'C4G','VLB','SAS']

# ------------------------------
# Crawl RSS
# ------------------------------
def crawl_rss(feeds, days_back=3, max_per_feed=50):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    rows = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for entry in d.entries[:max_per_feed]:
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                else:
                    published = datetime.now(timezone.utc)
                if published < cutoff:
                    continue
                link = entry.link
                downloaded = trafilatura.fetch_url(link)
                if not downloaded:
                    continue
                text = trafilatura.extract(downloaded)
                if not text:
                    continue
                rows.append({
                    "title": entry.title,
                    "published": published,
                    "link": link,
                    "text": text
                })
        except Exception as e:
            print(f"⚠️ Error parsing {url}: {e}")
    return pd.DataFrame(rows)

# ------------------------------
# Detect ticker in text
# ------------------------------
def detect_tickers(text, tickers=HSX_TICKERS):
    found = [t for t in tickers if f" {t} " in text.upper()]
    return list(set(found))

# ------------------------------
# Sentiment pipeline
# ------------------------------
def build_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment", use_fast=False)

# ------------------------------
# Analyze sentiment
# ------------------------------
def analyze(df, sentiment_model):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            txt = row["text"]
            if detect(txt) != "vi":
                continue
            tickers = detect_tickers(txt)
            if not tickers:
                continue
            sent = sentiment_model(txt[:512])[0]
            label = sent["label"].lower()
            score = sent["score"]
            for t in tickers:
                results.append({
                    "ticker": t,
                    "label": label,
                    "score": score,
                    "published": row["published"],
                    "link": row["link"]
                })
        except Exception as e:
            print(f"⚠️ Sentiment error: {e}")
    return pd.DataFrame(results)

# ------------------------------
# Dashboard
# ------------------------------
def visualize_dashboard(dash_df, outdir="out"):
    os.makedirs(outdir, exist_ok=True)
    if len(dash_df) == 0:
        print("⚠️ Dashboard rỗng, không có dữ liệu để vẽ.")
        return

    # --- Heatmap ---
    heatmap_data = dash_df.melt(
        id_vars=["ticker"],
        value_vars=["positive","neutral","negative"],
        var_name="sentiment",
        value_name="count"
    )

    plt.figure(figsize=(12,8))
    pivot = heatmap_data.pivot(index="ticker", columns="sentiment", values="count").fillna(0)
    sns.heatmap(pivot, annot=True, fmt="g", cmap="RdYlGn")
    plt.title("Heatmap Sentiment theo mã cổ phiếu")
    plt.tight_layout()
    plt.savefig(f"{outdir}/sentiment_heatmap.png", dpi=200)
    plt.close()

    # --- Pie chart ---
    total_pos = dash_df["positive"].sum()
    total_neu = dash_df["neutral"].sum()
    total_neg = dash_df["negative"].sum()
    plt.figure(figsize=(6,6))
    plt.pie([total_pos, total_neu, total_neg],
            labels=["Positive","Neutral","Negative"],
            autopct='%1.1f%%',
            colors=["#2ecc71","#f1c40f","#e74c3c"])
    plt.title("Tỉ lệ sentiment toàn bộ bài báo")
    plt.savefig(f"{outdir}/sentiment_pie.png", dpi=200)
    plt.close()

    # --- Plotly interactive ---
    fig = px.bar(
        dash_df.sort_values("articles", ascending=False).head(20),
        x="ticker", y=["positive","neutral","negative"],
        title="Top 20 mã được nhắc nhiều nhất và sentiment",
        barmode="stack"
    )
    fig.write_html(f"{outdir}/sentiment_dashboard.html")
    print(f"📊 Dashboard đã xuất ra thư mục {outdir}")

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=3)
    parser.add_argument("--max-per-feed", type=int, default=30)
    parser.add_argument("--outdir", type=str, default="out")
    args = parser.parse_args()

    feeds = [
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
        

        # International wires referencing Vietnam (optional)
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.bloomberg.com/feeds/podcasts.xml",
        "https://www.ft.com/?format=rss",
    ]

    print("📥 Crawling RSS...")
    df = crawl_rss(feeds, args.days_back, args.max_per_feed)
    print(f"✅ Collected {len(df)} articles")

    print("🤖 Loading sentiment model...")
    sentiment_model = build_sentiment_pipeline()

    print("🔎 Analyzing sentiment...")
    res = analyze(df, sentiment_model)
    res.to_csv(f"{args.outdir}/raw_results.csv", index=False)

    # --- Aggregate by ticker ---
    if len(res) > 0:
        dash = res.groupby(["ticker","label"]).size().unstack(fill_value=0)
        for col in ["positive","neutral","negative"]:
            if col not in dash.columns:
                dash[col] = 0
        dash["articles"] = dash.sum(axis=1)
        dash = dash.reset_index()
        dash.to_csv(f"{args.outdir}/summary_by_ticker.csv", index=False)
        visualize_dashboard(dash, args.outdir)
    else:
        print("⚠️ Không có kết quả sentiment hợp lệ.")

if __name__ == "__main__":
    main()
