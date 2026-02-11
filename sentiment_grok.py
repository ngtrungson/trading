# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:08:27 2025

@author: tson
"""

import requests
from bs4 import BeautifulSoup
from underthesea import sentiment
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Để vẽ heatmap đẹp hơn, nhưng nếu không có, có thể dùng plt.imshow

# Danh sách 100 mã cổ phiếu VN100 (cập nhật mới nhất dựa trên dữ liệu 2025)
stocks = [
    "ANV", "ASM", "BCG", "BCM", "BID", "BVH", "BWE", "CMG", "CII", "CTR", "CSM", "CTG", "DBC", "DBD", "DCM",
    "DGW", "DGC", "DHG", "DIG", "DPM", "DXG", "EIB", "FCN", "FIT", "FPT", "GAS", "GEG", "GEX", "GMD", "GVR",
    "HAH", "HBC", "HDB", "HDG", "HHP", "HPG", "HCM", "HSG", "HTN", "HVN", "IJC", "IMP", "ITA", "KBC", "KDH",
    "KOS", "LPB", "MBB", "MBC", "MSB", "MWG", "NLG", "NT2", "OCB", "PAN", "PDR", "PHR", "PLX", "PNJ", "POW",
    "PTB", "PVD", "PVT", "REE", "SAB", "SAM", "SBT", "SCR", "SHB", "SHS", "SIP", "SSI", "STB", "STK", "SZC",
    "SZL", "TCB", "TCM", "TCH", "TPB", "VCB", "VCG", "VCI", "VGC", "VHC", "VHM", "VIB", "VIC", "VJC", "VMD",
    "VNM", "VPB", "VPG", "VPI", "VRE", "VSC", "VSH"
]

# Danh sách trang báo phổ biến (top 5, bạn có thể thêm để lên 100)
sites = {
    'cafef': 'https://s.cafef.vn/TimKiem/{query}.chn',
    'vietstock': 'https://vietstock.vn/tim-kiem.htm?keyword={query}',
    'tinnhanhchungkhoan': 'https://www.tinnhanhchungkhoan.vn/search?q={query}',
    'vneconomy': 'https://vneconomy.vn/search.htm?keywords={query}',
    'vietnambiz': 'https://vietnambiz.vn/search?q={query}'
}

# Hàm lấy link bài báo từ trang tìm kiếm
def get_article_links(search_url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('http') and ('/post/' in href or '/tin-tuc/' in href or '/chung-khoan/' in href):  # Lọc link bài báo
                links.append(href)
        return list(set(links[:5]))  # Lấy top 5 link duy nhất
    except Exception as e:
        print(f"Lỗi khi lấy link: {e}")
        return []

# Hàm scrape nội dung bài báo
def get_article_text(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.text.strip() for p in paragraphs if p.text.strip())
        return text
    except Exception as e:
        print(f"Lỗi khi scrape {url}: {e}")
        return ""

# Phân tích sentiment cho từng cổ phiếu
results = {}
for stock in stocks:
    all_sentiments = []
    for site_name, search_template in sites.items():
        search_url = search_template.format(query=stock + ' chứng khoán')
        links = get_article_links(search_url)
        for link in links:
            text = get_article_text(link)
            if text:
                sent = sentiment(text)  # positive, negative, neutral
                all_sentiments.append(sent)
            time.sleep(2)  # Tránh bị chặn
        time.sleep(5)  # Nghỉ giữa các site
    
    if all_sentiments:
        positive = all_sentiments.count('positive') / len(all_sentiments)
        negative = all_sentiments.count('negative') / len(all_sentiments)
        neutral = all_sentiments.count('neutral') / len(all_sentiments)
        results[stock] = {'positive': positive, 'negative': negative, 'neutral': neutral}
        print(f"{stock}: Positive {positive*100:.2f}%, Negative {negative*100:.2f}%, Neutral {neutral*100:.2f}% (dựa trên {len(all_sentiments)} bài báo)")
    else:
        print(f"{stock}: Không tìm thấy dữ liệu")

# Lưu kết quả vào file CSV nếu cần
df = pd.DataFrame(results).T
df.to_csv('sentiment_vn100.csv', encoding='utf-8')
print("Kết quả đã lưu vào sentiment_vn100.csv")

# Bổ sung vẽ biểu đồ heatmap
# Nếu bạn chưa có seaborn, cài đặt: pip install seaborn
# Heatmap: Hàng là mã cổ phiếu, Cột là loại sentiment, Giá trị là phần trăm (0-100)
plt.figure(figsize=(8, 20))  # Kích thước cao để hiển thị hết 100 mã
sns.heatmap(df * 100, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=0.5)
plt.title('Sentiment Heatmap for VN100 Stocks (%)')
plt.ylabel('Stocks')
plt.xlabel('Sentiment Type')
plt.tight_layout()
plt.savefig('sentiment_heatmap.png')  # Lưu thành file PNG để xem
plt.show()  # Hiển thị biểu đồ nếu chạy trong môi trường hỗ trợ (như Jupyter hoặc IDE)
print("Biểu đồ heatmap đã được lưu vào 'sentiment_heatmap.png'")