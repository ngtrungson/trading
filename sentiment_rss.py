import os
import feedparser
import re
import pandas as pd
from datetime import datetime
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# =============================
# 1. Danh sách RSS báo VN (ví dụ, thêm dần lên 100 nguồn)
# =============================
VN_RSS_FEEDS = [
    "https://vnexpress.net/rss/kinh-doanh.rss",
    "https://cafef.vn/rss/tai-chinh.chn",
    "https://tuoitre.vn/rss/kinh-doanh.rss",
    "https://nld.com.vn/kinh-te.rss",
    "https://vietnamnet.vn/rss/kinh-doanh.rss",
    "https://bnews.vn/RssFeed.aspx?cid=4",
    "https://thanhnien.vn/rss/kinh-doanh.rss",
]

# =============================
# 2. Danh sách mã cổ phiếu VN (ví dụ, cần mở rộng thêm)
# =============================
VN_TICKERS = [
    "VCB","BID","CTG","TCB","VPB",
    "HPG","FPT","VNM","MWG","SSI",
    "GAS","PLX","VIC","VHM","NVL",
    "MBB","STB","VIB","HDB","SHB"
]

# =============================
# 3. Load sentiment model HuggingFace
# =============================
print("🔄 Đang load model sentiment HuggingFace ...")
classifier = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer=AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment", use_fast=False)
)

# Mapping nhãn -> sentiment
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# =============================
# 4. Crawl RSS + detect ticker
# =============================
def crawl_rss(feeds):
    articles = []
    for feed_url in feeds:
        try:
            d = feedparser.parse(feed_url)
            for entry in d.entries:
                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": feed_url
                })
        except Exception as e:
            print(f"⚠️ Lỗi khi crawl {feed_url}: {e}")
    return pd.DataFrame(articles)

def detect_tickers(text, tickers=VN_TICKERS):
    found = []
    for t in tickers:
        if re.search(rf"\b{t}\b", text):
            found.append(t)
    return found

# =============================
# 5. Phân tích sentiment
# =============================
def analyze_articles(df):
    results = []
    for _, row in df.iterrows():
        text = (row["title"] or "") + " " + (row["summary"] or "")
        tickers = detect_tickers(text)
        if not tickers:
            continue
        try:
            sentiment = classifier(text[:512])[0]
            raw_label = sentiment["label"]
            label = label_map.get(raw_label, raw_label)
            score = sentiment["score"]
        except Exception as e:
            label, score = "neutral", 0.0

        results.append({
            "title": row["title"],
            "ticker": ",".join(tickers),
            "sentiment": label,
            "score": score,
            "link": row["link"],
            "published": row["published"],
            "source": row["source"]
        })
    return pd.DataFrame(results)

# =============================
# 6. Tóm tắt theo ticker
# =============================
def summarize_by_ticker(df):
    summary = df.groupby(["ticker","sentiment"]).size().unstack(fill_value=0).reset_index()
    # Đảm bảo đủ 3 cột
    for col in ["positive","neutral","negative"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["articles"] = summary[["positive","neutral","negative"]].sum(axis=1)
    return summary

# =============================
# 7. Visualization
# =============================
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

# =============================
# 8. Run all
# =============================
if __name__ == "__main__":
    print("🚀 Bắt đầu crawl RSS...")
    articles_df = crawl_rss(VN_RSS_FEEDS)
    print(f"✅ Crawl được {len(articles_df)} bài báo.")

    print("🔎 Đang phân tích sentiment...")
    analyzed_df = analyze_articles(articles_df)
    analyzed_df.to_csv("out/articles_sentiment.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Lưu chi tiết sentiment vào out/articles_sentiment.csv")

    summary_df = summarize_by_ticker(analyzed_df)
    summary_df.to_csv("out/summary_sentiment.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Lưu tổng hợp sentiment vào out/summary_sentiment.csv")

    visualize_dashboard(summary_df, outdir="out")
