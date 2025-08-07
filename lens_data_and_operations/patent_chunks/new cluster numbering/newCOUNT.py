
import pandas as pd

# خواندن فایل ورودی
df = pd.read_csv("sigma_input.csv")

# مرتب‌سازی مطمئن برای درست بودن تجمع
df = df.sort_values(['Cluster ID', 'Year'])

# محاسبه مقدار تجمعی در هر کلاستر
df['Cumulative Count'] = df.groupby('Cluster ID')['Count'].cumsum()

# فقط ستون‌های موردنظر رو نگه می‌داریم
output_df = df[['Cluster ID', 'Cluster Name', 'Year', 'Cumulative Count']]

# ذخیره فایل خروجی
output_df.to_csv("sigma_input_cumulative.csv", index=False)

print("✅ فایل جدید با مقدار تجمعی ذخیره شد: sigma_input_cumulative.csv")
