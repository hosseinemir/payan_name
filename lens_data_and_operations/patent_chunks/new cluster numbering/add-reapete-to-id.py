import pandas as pd
import ast

# فایل‌ها
file_repet = "word-weight-repet.csv"
file_info = "Id,modularity_class,strength.csv"
output_file = "id_class_strength_fullrepet.csv"

# بارگیری فایل تکرار واژه‌ها
df_repet = pd.read_csv(file_repet)
df_repet["word"] = df_repet["word"].astype(str)

# تبدیل رشته repet به لیست پایتونی
df_repet["repet"] = df_repet["repet"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# ساخت دیکشنری: word → full repet list
repet_dict = dict(zip(df_repet["word"], df_repet["repet"]))

# بارگیری فایل دوم
df_info = pd.read_csv(file_info)
df_info["Id"] = df_info["Id"].astype(str)

# اضافه کردن لیست کامل تکرار به هر Id
df_info["full_repet"] = df_info["Id"].apply(lambda x: repet_dict.get(x, []))

# ذخیره فایل نهایی
df_info.to_csv(output_file, index=False)

print("✅ فایل خروجی ساخته شد با ستون full_repet:", output_file)
