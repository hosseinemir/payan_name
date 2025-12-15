import os
import csv
import time
import pickle
from collections import defaultdict, Counter
from itertools import combinations

# =====================================================
# GLOBAL FACTORY (pickle-safe)
# =====================================================
def counter_factory():
    return Counter()

# =====================================================
# CONFIG
# =====================================================
INPUT_FILES = [
    "ev_articles_part1.csv",
    "ev_articles_part2.csv",
]

CHUNK_SIZE = 5000

TMP_DIR = "tmp_processing"
WORD_TMP = os.path.join(TMP_DIR, "word_stats.pkl")
WORD_YEAR_TMP = os.path.join(TMP_DIR, "word_year_stats.pkl")
EDGE_TMP = os.path.join(TMP_DIR, "edge_stats.pkl")
PROGRESS_FILE = os.path.join(TMP_DIR, "progress.pkl")

OUTPUT_WORD_FILE = "word_year_stats.csv"
OUTPUT_EDGE_FILE = "co_occurrence_edges.csv"

# =====================================================
# UTILS
# =====================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_tmp():
    os.makedirs(TMP_DIR, exist_ok=True)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path, default):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default

# =====================================================
# MAIN PROCESS
# =====================================================
def process_files():
    ensure_tmp()

    log("Loading progress and temporary data...")
    progress = load_pickle(PROGRESS_FILE, {"file_idx": 0, "row_idx": 0})

    word_total = load_pickle(WORD_TMP, Counter())
    word_year = load_pickle(
        WORD_YEAR_TMP,
        defaultdict(counter_factory)
    )
    edge_counter = load_pickle(EDGE_TMP, Counter())

    for f_idx in range(progress["file_idx"], len(INPUT_FILES)):
        file_path = INPUT_FILES[f_idx]
        log(f"Processing file: {file_path}")

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            buffer = []

            for r_idx, row in enumerate(reader):
                if f_idx == progress["file_idx"] and r_idx < progress["row_idx"]:
                    continue

                buffer.append(row)

                if len(buffer) >= CHUNK_SIZE:
                    process_chunk(buffer, word_total, word_year, edge_counter)
                    buffer.clear()

                    progress = {"file_idx": f_idx, "row_idx": r_idx + 1}
                    checkpoint(word_total, word_year, edge_counter, progress)

            if buffer:
                process_chunk(buffer, word_total, word_year, edge_counter)
                buffer.clear()

                progress = {"file_idx": f_idx + 1, "row_idx": 0}
                checkpoint(word_total, word_year, edge_counter, progress)

    log("All files processed.")
    finalize_outputs(word_total, word_year, edge_counter)
    cleanup()

# =====================================================
# CHUNK PROCESSING
# =====================================================
def process_chunk(rows, word_total, word_year, edge_counter):
    for row in rows:
        year = str(row["Year"])
        tokens = row["title_abstract"].split()

        # -------- WORD FREQUENCY --------
        for w in tokens:
            word_total[w] += 1
            word_year[w][year] += 1

        # -------- CO-OCCURRENCE --------
        unique_tokens = sorted(set(tokens))
        for a, b in combinations(unique_tokens, 2):
            edge_counter[(a, b)] += 1

# =====================================================
# CHECKPOINT
# =====================================================
def checkpoint(word_total, word_year, edge_counter, progress):
    log("Checkpoint: saving intermediate results...")
    save_pickle(word_total, WORD_TMP)
    save_pickle(word_year, WORD_YEAR_TMP)
    save_pickle(edge_counter, EDGE_TMP)
    save_pickle(progress, PROGRESS_FILE)

# =====================================================
# OUTPUT
# =====================================================
def finalize_outputs(word_total, word_year, edge_counter):
    log("Writing word-year statistics output...")
    with open(OUTPUT_WORD_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "weight", "year"])

        for word, total in word_total.items():
            year_dict = word_year[word]
            year_str = "[" + ",".join(
                f"(\"{y}\":\"{c}\")"
                for y, c in sorted(year_dict.items())
            ) + "]"
            writer.writerow([word, total, year_str])

    log("Writing co-occurrence edge output...")
    with open(OUTPUT_EDGE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])

        for (a, b), c in edge_counter.items():
            writer.writerow([a, b, c])

# =====================================================
# CLEANUP
# =====================================================
def cleanup():
    log("Cleaning temporary files...")
    for file in [
        WORD_TMP,
        WORD_YEAR_TMP,
        EDGE_TMP,
        PROGRESS_FILE,
    ]:
        if os.path.exists(file):
            os.remove(file)

    if os.path.exists(TMP_DIR) and not os.listdir(TMP_DIR):
        os.rmdir(TMP_DIR)

    log("Cleanup completed.")

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    start_time = time.time()
    log("Pipeline started.")
    process_files()
    log(f"Total execution time: {round(time.time() - start_time, 2)} seconds")
