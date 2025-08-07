import pandas as pd
from tqdm import tqdm

# خواندن فایل‌ها
df1 = pd.read_csv("Id_modularity_class_strength.csv")
df2 = pd.read_csv("word-weight-repet.csv")

# استخراج ستون‌ها
ids = set(df1['Id'].astype(str))
words = df2['word'].astype(str)

# استفاده از tqdm برای نمایش پیشرفت
tqdm.pandas(desc="Checking reverse differences")

# بررسی اینکه آیا هر word در id هست یا نه
diff_words = words[~words.progress_apply(lambda x: x in ids)]

# خروجی گرفتن
diff_words.to_csv("diff_word_2.csv", index=False, header=True)

print("✅ کلمات متفاوت در فایل diff_word_2.csv ذخیره شدند.")
