import pandas as pd
import ast

# خواندن فایل top30withName و نادیده گرفتن ردیف سوم
top30_df = pd.read_csv("NameAndNumber.csv", skiprows=[2])
top30_df.columns = top30_df.columns.str.strip()  # حذف فاصله اضافی
top30_df["modularity_class"] = top30_df["modularity_class"].astype(int)

# خواندن فایل دوم
data_df = pd.read_csv("id_class_strength_fullrepet.csv")
data_df.columns = data_df.columns.str.strip()  # حذف فاصله اضافی
data_df["modularity_class"] = data_df["modularity_class"].astype(int)

# ساخت دیتافریم خروجی
output_rows = []

for _, row in top30_df.iterrows():
    cluster_id = row["modularity_class"]
    cluster_name = row["Technology_Domain_Name"]

    # فیلتر رکوردهای متعلق به این کلاستر
    cluster_data = data_df[data_df["modularity_class"] == cluster_id]

    year_counts = {}

    for _, entry in cluster_data.iterrows():
        full_repet = entry["full_repet"]
        try:
            year_data = ast.literal_eval(full_repet)
            for year, count in year_data:
                year_counts[year] = year_counts.get(year, 0) + count
        except Exception:
            continue  # اگر داده full_repet معیوب بود، ازش عبور کن

    # ساخت ردیف‌های خروجی برای این کلاستر
    for year in sorted(year_counts):
        output_rows.append({
            "Cluster ID": cluster_id,
            "Cluster Name": cluster_name,
            "Year": year,
            "Count": year_counts[year]
        })

# ساخت دیتافریم خروجی
output_df = pd.DataFrame(output_rows)

# مرتب‌سازی نهایی
output_df.sort_values(by=["Cluster ID", "Year"], inplace=True)

# ذخیره خروجی
output_df.to_csv("sigma_input.csv", index=False)
print("✅ فایل sigma_input.csv با موفقیت ساخته شد.")
