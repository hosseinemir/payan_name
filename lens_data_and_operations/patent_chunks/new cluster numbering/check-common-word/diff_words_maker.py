import pandas as pd
from tqdm import tqdm

# خواندن فایل‌ها
df1 = pd.read_csv("Id_modularity_class_strength.csv")
df2 = pd.read_csv("word-weight-repet.csv")

# استخراج ستون‌ها
ids = df1['Id'].astype(str)
words = set(df2['word'].astype(str))  # استفاده از set برای سریع‌تر شدن مقایسه

# استفاده از tqdm برای نمایش پیشرفت
tqdm.pandas(desc="Checking differences")

# بررسی اینکه آیا هر id در فایل دوم هست یا نه
diff_ids = ids[~ids.progress_apply(lambda x: x in words)]

# خروجی گرفتن
diff_ids.to_csv("diff_word.csv", index=False, header=True)

print("✅ مقادیر متفاوت در فایل diff_word.csv ذخیره شدند.")
