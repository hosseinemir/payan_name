import pandas as pd

# مسیر فایل ورودی
input_file = "sigma_input_cumulative.csv"

# خواندن فایل
df = pd.read_csv(input_file)

# حذف ردیف‌هایی که Year برابر با 2025 هستند
df_filtered = df[df["Year"] != 2025]

# ذخیره فایل خروجی
output_file = "sigma_input_cumulative_no2025.csv"
df_filtered.to_csv(output_file, index=False)

print(f"✅ فایل بدون سال 2025 ذخیره شد به نام: {output_file}")
