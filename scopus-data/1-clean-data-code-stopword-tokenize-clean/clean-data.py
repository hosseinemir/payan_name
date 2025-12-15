import pandas as pd
from pathlib import Path
import re

# اگر nltk را قبلاً نصب نکردی:
# pip install nltk

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# =========================
#  1. آماده‌سازی ابزارهای NLP
# =========================

# دانلود stopwords در اولین اجرا
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """lowercase, حذف کاراکترهای غیرحرفی، حذف stopwords، و stemming"""
    if not isinstance(text, str):
        return ""

    # همه حروف کوچک
    text = text.lower()

    # حذف هر چیزی غیر از حروف و فاصله
    text = re.sub(r'[^a-z\s]', ' ', text)

    # توکن‌سازی ساده با split
    tokens = text.split()

    # حذف stopwords و اعمال stemming
    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(cleaned_tokens)

# =========================
#  2. خواندن همه CSV ها از فولدر جاری
# =========================

folder = Path(".")
csv_files = sorted(folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("هیچ فایل CSV در فولدر جاری پیدا نشد.")

dfs = []
for file in csv_files:
    print(f"Reading: {file.name}")
    df_tmp = pd.read_csv(file)
    dfs.append(df_tmp)

# ادغام همه دیتافریم‌ها
df = pd.concat(dfs, ignore_index=True)

# =========================
#  3. نگه داشتن فقط ستون‌های مورد نیاز
# =========================

required_cols = ["Title", "Year", "Abstract"]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"ستون‌های زیر در داده‌ها پیدا نشدند: {missing_cols}")

df = df[required_cols].copy()

# =========================
#  4. حذف ردیف‌های بدون Title یا Abstract
# =========================

# حذف NaN
df = df.dropna(subset=["Title", "Abstract"])

# حذف فضاهای اضافی
df["Title"] = df["Title"].astype(str).str.strip()
df["Abstract"] = df["Abstract"].astype(str).str.strip()

# حذف ردیف‌هایی که بعد از strip خالی‌اند
df = df[(df["Title"] != "") & (df["Abstract"] != "")].copy()

# =========================
#  5. ساخت ستون ترکیبی Title + Abstract
# =========================

df["title_abstract_raw"] = df["Title"] + ". " + df["Abstract"]

# =========================
#  6. پاک‌سازی متنی روی ستون ترکیبی
# =========================

df["title_abstract"] = df["title_abstract_raw"].apply(clean_text)

# حذف ردیف‌هایی که بعد از پاکسازی متن خالی شده‌اند
df = df[df["title_abstract"].str.strip() != ""].copy()

# =========================
#  7. آماده‌سازی دیتافریم نهایی (فقط Year و متن ترکیبی)
# =========================

final_df = df[["Year", "title_abstract"]].copy()

# در صورت نیاز Year را به int تبدیل کن (اختیاری):
# final_df["Year"] = pd.to_numeric(final_df["Year"], errors="coerce")
# final_df = final_df.dropna(subset=["Year"])
# final_df["Year"] = final_df["Year"].astype(int)

# =========================
#  8. تقسیم به دو فایل با تعداد ردیف مساوی
# =========================

n = len(final_df)
if n < 2:
    raise ValueError("بعد از پاکسازی، تعداد ردیف‌ها کمتر از ۲ است. داده کافی برای تقسیم به دو فایل وجود ندارد.")

# اگر فرد باشد، یک ردیف آخر حذف می‌شود تا زوج شود
if n % 2 != 0:
    print("تعداد ردیف‌ها فرد است، آخرین ردیف برای برابری دو فایل حذف می‌شود.")
    final_df = final_df.iloc[:-1, :]
    n = len(final_df)

half = n // 2

df_part1 = final_df.iloc[:half].reset_index(drop=True)
df_part2 = final_df.iloc[half:].reset_index(drop=True)

# =========================
#  9. ذخیره خروجی‌ها
# =========================

output_file_1 = "ev_articles_part1.csv"
output_file_2 = "ev_articles_part2.csv"

df_part1.to_csv(output_file_1, index=False)
df_part2.to_csv(output_file_2, index=False)

print(f"Saved: {output_file_1}  (rows: {len(df_part1)})")
print(f"Saved: {output_file_2}  (rows: {len(df_part2)})")
