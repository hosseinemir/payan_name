import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ثابت نگه‌داشتن تصادفی‌سازی
np.random.seed(42)

# ------------------------------
# 1) تابع لجستیک (مثل قبل - استفاده نمی‌شود اما حفظ شده)
# ------------------------------
def logistic_function(t, k, a, b):
    """
    Logistic function for S-curve fitting
    
    Parameters:
    t: time (years)
    k: asymptotic saturation level (maximum expected number of patents)
    a: inflection point (year of fastest growth)
    b: controls the growth rate's steepness
    """
    return k / (1 + np.exp(-(t - a) / b))

# ------------------------------
# 2) لایه‌ی تطبیق با ورودی شما
# ------------------------------
def load_and_adapt_user_data(user_csv_path):
    """
    Reads user's cumulative input:
      Columns: Cluster ID, Cluster Name, Year, Cumulative Count
    Converts to annual counts per cluster and returns a normalized DataFrame:
      Columns: Topic (Cluster ID), Publication Year, count
    """
    df = pd.read_csv(user_csv_path)
    df.columns = [c.strip() for c in df.columns]
    required = {"Cluster ID", "Cluster Name", "Year", "Cumulative Count"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    gdf = df.rename(columns={
        "Cluster ID": "cluster_id",
        "Cluster Name": "cluster_name",
        "Year": "year",
        "Cumulative Count": "cum"
    }).copy()

    # types
    gdf["year"] = pd.to_numeric(gdf["year"], errors="coerce").astype("Int64")
    gdf["cum"]  = pd.to_numeric(gdf["cum"], errors="coerce").astype(float)
    gdf = gdf.dropna(subset=["cluster_id", "year", "cum"]).copy()
    gdf["year"] = gdf["year"].astype(int)

    # مرتب‌سازی و یکنواخت‌سازی تجمعی (اگر احیاناً نزولی باشد)
    gdf = gdf.sort_values(["cluster_id", "year"])
    gdf["cum"] = gdf.groupby("cluster_id")["cum"].cummax()

    # تبدیل تجمعی به سالانه
    gdf["annual"] = gdf.groupby("cluster_id")["cum"].diff().fillna(gdf["cum"])
    gdf["annual"] = gdf["annual"].clip(lower=0).round().astype(int)
    gdf = gdf[gdf["annual"] > 0].copy()

    # Topic = Cluster ID (عدد اگر ممکن بود)
    try:
        topic_vals = gdf["cluster_id"].astype(int)
    except Exception:
        topic_vals = gdf["cluster_id"].astype(str)

    out = pd.DataFrame({
        "Topic": topic_vals,
        "Publication Year": gdf["year"].astype(int),
        "count": gdf["annual"].astype(int)
    })

    return out

# ------------------------------
# 3) تحلیل ویژگی‌های تکنولوژی (با پشتیبانی از count)
# ------------------------------
def analyze_technology_characteristics(topic_data, year_column='Publication Year'):
    """
    Analyze key characteristics of a technology to determine its maturity stage.
    Supports two input shapes:
      - raw rows (no 'count' col): uses .size()
      - aggregated rows (has 'count'): uses sum(count)
    """
    if "count" in topic_data.columns:
        annual_counts = (topic_data.groupby(year_column)["count"]
                         .sum().reset_index(name='count'))
        total_patents = int(annual_counts["count"].sum())
    else:
        annual_counts = topic_data.groupby(year_column).size().reset_index(name='count')
        total_patents = len(topic_data)

    annual_counts = annual_counts.sort_values(year_column)
    years = annual_counts[year_column].values
    counts = annual_counts['count'].values

    # key metrics
    year_span = int(years.max() - years.min()) if len(years) else 0
    avg_annual = total_patents / max(year_span, 1) if year_span > 0 else total_patents

    # recent activity (>= 2022)
    recent_years = annual_counts[annual_counts[year_column] >= 2022]
    recent_activity = int(recent_years['count'].sum()) if len(recent_years) > 0 else 0
    recent_ratio = (recent_activity / total_patents) if total_patents > 0 else 0.0

    # peak year
    peak_year = int(years[np.argmax(counts)]) if len(counts) else None

    # periods
    early_period = int(annual_counts[annual_counts[year_column] <= 2010]['count'].sum())
    middle_period = int(annual_counts[(annual_counts[year_column] > 2010) & 
                                      (annual_counts[year_column] <= 2018)]['count'].sum())
    late_period = int(annual_counts[annual_counts[year_column] > 2018]['count'].sum())

    # recent trend (3 last years)
    if len(counts) >= 3:
        # use last 3 distinct years
        yrs_tail = years[-3:]
        cnt_tail = counts[-3:]
        # ensure unique years
        if len(np.unique(yrs_tail)) >= 2:
            recent_trend = np.polyfit(yrs_tail, cnt_tail, 1)[0]
        else:
            recent_trend = 0.0
    else:
        recent_trend = 0.0

    return {
        'total_patents': total_patents,
        'year_span': year_span,
        'avg_annual': avg_annual,
        'recent_ratio': recent_ratio,
        'peak_year': peak_year,
        'recent_trend': recent_trend,
        'early_activity': (early_period / total_patents) if total_patents>0 else 0.0,
        'middle_activity': (middle_period / total_patents) if total_patents>0 else 0.0,
        'late_activity': (late_period / total_patents) if total_patents>0 else 0.0
    }

# ------------------------------
# 4) طبقه‌بندی مرحله بلوغ (مثل قبل)
# ------------------------------
def classify_maturity_stage(characteristics):
    """
    Classify technology maturity based on multiple characteristics
    Returns maturity year in one of four categories:
    - Already Mature: ≤ 2024
    - Near-term: 2025-2027
    - Medium-term: 2028-2030  
    - Long-term: 2031+
    """
    total_patents   = characteristics['total_patents']
    recent_ratio    = characteristics['recent_ratio']
    peak_year       = characteristics['peak_year']
    recent_trend    = characteristics['recent_trend']
    early_activity  = characteristics['early_activity']
    late_activity   = characteristics['late_activity']

    maturity_score = 0

    # size
    if total_patents > 1000: maturity_score += 3
    elif total_patents > 500: maturity_score += 2
    elif total_patents > 200: maturity_score += 1

    # historical
    if early_activity > 0.4: maturity_score += 3
    elif early_activity > 0.2: maturity_score += 2

    # peak timing
    if peak_year is not None:
        if peak_year < 2015: maturity_score += 3
        elif peak_year < 2020: maturity_score += 2
        elif peak_year < 2022: maturity_score += 1

    # recent activity (inverse)
    if recent_ratio < 0.2: maturity_score += 2
    elif recent_ratio < 0.4: maturity_score += 1
    elif recent_ratio > 0.6: maturity_score -= 1

    # recent trend
    if recent_trend < -10: maturity_score += 2
    elif recent_trend < 0: maturity_score += 1
    elif recent_trend > 20: maturity_score -= 2
    elif recent_trend > 5: maturity_score -= 1

    random_factor = np.random.uniform(-0.5, 0.5)
    final_score = maturity_score + random_factor

    if final_score >= 6:
        maturity_year = np.random.choice([2022, 2023, 2024])
        stage = "Already Mature"
    elif final_score >= 4:
        maturity_year = np.random.choice([2025, 2026, 2027])
        stage = "Near-term"
    elif final_score >= 1:
        maturity_year = np.random.choice([2028, 2029, 2030])
        stage = "Medium-term"
    else:
        if final_score > -2:
            maturity_year = np.random.choice([2031, 2032, 2033])
            stage = "Long-term"
        else:
            maturity_year = None
            stage = "Still Emerging"

    return maturity_year, stage, final_score

# ------------------------------
# 5) برآورد مراحل چرخه‌عمر (با پشتیبانی از count)
# ------------------------------
def estimate_lifecycle_stages(df, topic_column='Topic', year_column='Publication Year', min_years=4):
    """
    Estimate technology lifecycle stages using characteristic-based classification.
    Works for:
      - exploded rows (no 'count')
      - aggregated rows (has 'count' per Topic-Year)
    """
    results = []
    topics = df[topic_column].unique()

    for topic in topics:
        print(f"Processing topic: {topic}")

        topic_data = df[df[topic_column] == topic].copy()

        # Aggregate annual patent counts
        if "count" in topic_data.columns:
            annual_counts = (topic_data.groupby(year_column)["count"]
                             .sum().reset_index(name='count'))
        else:
            annual_counts = topic_data.groupby(year_column).size().reset_index(name='count')

        annual_counts = annual_counts.sort_values(year_column)

        # enough distinct years?
        if len(annual_counts) < min_years:
            print(f"  Insufficient data ({len(annual_counts)} years) - skipping")
            results.append({
                'Technology Theme': topic,
                'Emerging': None,
                'Growth': None,
                'Maturity': None,
                'Saturation': None,
                'Stage_Category': 'Insufficient_Data',
                'Maturity_Score': None
            })
            continue

        # analyze + classify
        characteristics = analyze_technology_characteristics(topic_data, year_column)
        maturity_year, stage_category, maturity_score = classify_maturity_stage(characteristics)

        # other lifecycle stages based on maturity
        if maturity_year:
            emerging_offset = np.random.randint(3, 9)
            emerging_year = max(1990, maturity_year - emerging_offset)

            growth_offset = np.random.randint(1, 4)
            growth_year = max(emerging_year + 1, maturity_year - growth_offset)

            saturation_year = maturity_year + 5
        else:
            # emerging technologies
            emerging_year = max(2018, 2025 - np.random.randint(2, 6))
            growth_year = emerging_year + np.random.randint(2, 4)
            maturity_year = None
            saturation_year = None

        results.append({
            'Technology Theme': topic,
            'Emerging': emerging_year,
            'Growth': growth_year,
            'Maturity': maturity_year,
            'Saturation': saturation_year,
            'Stage_Category': stage_category,
            'Maturity_Score': round(maturity_score, 2) if maturity_score is not None else None
        })

        print(f"  Classification: {stage_category} (Score: {maturity_score:.2f})")

    return pd.DataFrame(results)

# ------------------------------
# 6) ترسیم نمودارها (با پشتیبانی از count)
# ------------------------------
def plot_lifecycle_curves(df, topic_column='Topic', year_column='Publication Year', 
                         topics_to_plot=None, max_plots=6):
    """
    Plot patent activity over time for selected topics.
    Works with aggregated data (has 'count') or raw rows.
    """
    topics = df[topic_column].unique()
    if topics_to_plot is None:
        topics_to_plot = topics[:max_plots]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, topic in enumerate(topics_to_plot):
        if i >= len(axes):
            break

        topic_data = df[df[topic_column] == topic].copy()

        if "count" in topic_data.columns:
            annual_counts = (topic_data.groupby(year_column)["count"]
                             .sum().reset_index(name='count'))
        else:
            annual_counts = topic_data.groupby(year_column).size().reset_index(name='count')

        annual_counts = annual_counts.sort_values(year_column)
        if len(annual_counts) < 4:
            continue

        years = annual_counts[year_column].values
        counts = annual_counts['count'].values
        cumulative = annual_counts['count'].cumsum().values

        axes[i].bar(years, counts, alpha=0.6, label='Annual Patents', color='lightblue')
        ax2 = axes[i].twinx()
        ax2.plot(years, cumulative, 'r-', linewidth=2, label='Cumulative Patents')

        axes[i].set_title(f'Topic {topic}', fontsize=10)
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Annual Patents', color='blue')
        ax2.set_ylabel('Cumulative Patents', color='red')
        axes[i].grid(True, alpha=0.3)

        lines1, labels1 = axes[i].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[i].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    for i in range(len(topics_to_plot), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

# ------------------------------
# 7) Main
# ------------------------------
if __name__ == "__main__":
    # مسیر فایل شما در کنار کد
    user_file_path = "./sigma_input_cumulative_no2025.csv"

    try:
        print("Loading and adapting user cumulative data...")
        df_norm = load_and_adapt_user_data(user_file_path)
        print(f"Data adapted. Shape: {df_norm.shape}  (columns: {list(df_norm.columns)})")

        # نمایش اطلاعات پایه
        n_topics = df_norm['Topic'].nunique()
        year_min = df_norm['Publication Year'].min()
        year_max = df_norm['Publication Year'].max()

        # تعداد پتنت هر Topic (با توجه به count)
        topic_counts = (df_norm.groupby('Topic')['count'].sum()
                        if 'count' in df_norm.columns
                        else df_norm.groupby('Topic').size())
        print(f"\nTopics found: {n_topics}")
        print(f"Year range: {year_min} - {year_max}")
        print(f"Topic distribution:")
        for topic, count in topic_counts.sort_index().items():
            print(f"  Topic {topic}: {int(count)} patents")

        # برآورد مراحل
        print("\nEstimating lifecycle stages...")
        lifecycle_results = estimate_lifecycle_stages(df_norm)

        # نمایش نتایج
        print("\n" + "="*80)
        print("TECHNOLOGY LIFECYCLE STAGE ESTIMATION RESULTS")
        print("="*80)

        display_results = lifecycle_results.copy()
        display_results['Maturity_sort'] = display_results['Maturity'].fillna(9999)
        display_results = display_results.sort_values('Maturity_sort').drop('Maturity_sort', axis=1)

        for _, row in display_results.iterrows():
            print(f"\nTopic {row['Technology Theme']} ({row['Stage_Category']}):")
            print(f"  Emerging: {row['Emerging'] if pd.notna(row['Emerging']) else '–'}")
            print(f"  Growth: {row['Growth'] if pd.notna(row['Growth']) else '–'}")
            print(f"  Maturity: {row['Maturity'] if pd.notna(row['Maturity']) else '–'}")
            print(f"  Saturation: {row['Saturation'] if pd.notna(row['Saturation']) else '–'}")
            print(f"  Maturity Score: {row['Maturity_Score'] if pd.notna(row['Maturity_Score']) else '–'}")

        # خلاصه دسته‌ها
        print(f"\n{'='*80}")
        print("MATURITY TIMELINE CATEGORIES")
        print(f"{'='*80}")

        categories = {
            'Already Mature (≤2024)': [],
            'Near-term (2025-2027)': [],
            'Medium-term (2028-2030)': [],
            'Long-term (2031+)': [],
            'Still Emerging': []
        }

        for _, row in display_results.iterrows():
            topic = str(row['Technology Theme'])
            stage_category = row['Stage_Category']

            if stage_category == 'Already Mature':
                categories['Already Mature (≤2024)'].append(topic)
            elif stage_category == 'Near-term':
                categories['Near-term (2025-2027)'].append(topic)
            elif stage_category == 'Medium-term':
                categories['Medium-term (2028-2030)'].append(topic)
            elif stage_category == 'Long-term':
                categories['Long-term (2031+)'].append(topic)
            elif stage_category == 'Still Emerging':
                categories['Still Emerging'].append(topic)
            else:
                if 'Insufficient_Data' in str(stage_category):
                    categories['Still Emerging'].append(topic)
                else:
                    print(f"Warning: Unexpected category '{stage_category}' for topic {topic}")
                    categories['Still Emerging'].append(topic)

        for category, tpcs in categories.items():
            print(f"\n{category}: {len(tpcs)} topics")
            if tpcs:
                print(f"  {', '.join(tpcs)}")

        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        print(f"{'Topic':<8} {'Category':<15} {'Emerging':<10} {'Growth':<8} {'Maturity':<10} {'Saturation':<12} {'Score':<8}")
        print(f"{'-'*8} {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*12} {'-'*8}")

        for _, row in display_results.iterrows():
            emerging = str(int(row['Emerging'])) if pd.notna(row['Emerging']) else '–'
            growth = str(int(row['Growth'])) if pd.notna(row['Growth']) else '–'
            maturity = str(int(row['Maturity'])) if pd.notna(row['Maturity']) else '–'
            saturation = str(int(row['Saturation'])) if pd.notna(row['Saturation']) else '–'
            score = str(row['Maturity_Score']) if pd.notna(row['Maturity_Score']) else '–'
            category_short = row['Stage_Category'].replace('Already Mature', 'Mature').replace('Medium-term', 'Med-term')[:12]

            print(f"{str(row['Technology Theme']):<8} {category_short:<15} {emerging:<10} {growth:<8} {maturity:<10} {saturation:<12} {score:<8}")

        # ذخیره خروجی
        output_file = "technology_lifecycle_stages.csv"
        lifecycle_results.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nResults saved to: {output_file}")

        # نمودارها: 6 تاپیکِ پر داده
        print("\nGenerating visualizations...")
        topic_sizes = (df_norm.groupby('Topic')['count'].sum()
                       if 'count' in df_norm.columns
                       else df_norm.groupby('Topic').size())
        top_topics = topic_sizes.sort_values(ascending=False).head(6).index.tolist()
        plot_lifecycle_curves(df_norm, topics_to_plot=top_topics)

    except FileNotFoundError:
        print(f"Error: Could not find the file at {user_file_path}")
        print("Please check the file path and make sure the file exists.")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your data format and file path.")
