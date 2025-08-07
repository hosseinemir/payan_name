import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os
import sys

# تنظیمات
OUTPUT_FILE = "word-weight-repet.csv"
NUM_FILES = 4
CHUNKSIZE = 5000  # بر اساس رم قابل تغییره

# ساختار دیکشنری نهایی: {word: {"weight": int, "repet": {year: count}}}
word_data = defaultdict(lambda: {"weight": 0, "repet": defaultdict(int)})

print("🚀 شروع پردازش فایل‌ها...")

try:
    for i in range(1, NUM_FILES + 1):
        filename = f"patents_part_{i}.csv"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"❌ فایل پیدا نشد: {filename}")

        print(f"📂 در حال پردازش فایل: {filename}")
        for chunk in tqdm(pd.read_csv(filename, chunksize=CHUNKSIZE), desc=f"📄 پردازش {filename}"):
            for row in chunk.itertuples(index=False):
                try:
                    text = str(row[1]).lower()
                    year = int(row[2])
                    words = text.split()

                    for word in words:
                        word_data[word]["weight"] += 1
                        word_data[word]["repet"][year] += 1

                except Exception as row_err:
                    print(f"⚠️ خطا در ردیف: {row} → {row_err}")
                    continue

        print(f"✅ فایل {filename} با موفقیت پردازش شد.\n")

    print("📦 ساخت دیتافریم خروجی...")

    output_rows = []
    for word, data in word_data.items():
        weight = data["weight"]
        if weight < 1:
            continue
        year_dist = sorted(data["repet"].items())
        output_rows.append({
            "word": word,
            "weight": weight,
            "repet": str(year_dist)
        })

    df_out = pd.DataFrame(output_rows)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"🎯 فایل خروجی ساخته شد: {OUTPUT_FILE}")

except Exception as e:
    print("💥 خطای کلی:")
    print(e)
    sys.exit(1)
