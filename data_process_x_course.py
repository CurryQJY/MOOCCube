import argparse
import heapq
import json
import os
import re
import tempfile
import zlib
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build course-level processed data for MOOCCubeX with the same layout as processed_data/."
    )
    parser.add_argument("--base-dir", default="./MOOCCubeX")
    parser.add_argument("--output-dir", default="./processed_data_x_course")
    parser.add_argument("--inter-file", default=None)
    parser.add_argument("--bucket-count", type=int, default=64)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--min-user-inter", type=int, default=1)
    parser.add_argument("--min-item-inter", type=int, default=1)
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--bert-model", default="bert-base-chinese")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def read_json_flexible(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def normalize_course_id(raw_course_id):
    if raw_course_id is None:
        return None
    cid = str(raw_course_id).strip()
    if not cid:
        return None
    if cid.startswith("C_"):
        return cid
    return f"C_{cid}"


def parse_enroll_time(raw_time):
    if not raw_time:
        return None
    try:
        dt = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return int(dt.timestamp())


def clean_course_text(course_obj):
    name = course_obj.get("name", "") or ""
    about = course_obj.get("about", "") or course_obj.get("intro", "") or ""
    about = re.sub(r"<[^>]+>", "", about)
    return f"{name} {about}".strip() or name or "Unknown Course"


class MOOCCubeXCourseProcessor:
    def __init__(self, args):
        self.args = args
        self.base_dir = args.base_dir
        self.output_dir = args.output_dir
        self.user_file = os.path.join(self.base_dir, "entities", "user.json")
        self.course_file = os.path.join(self.base_dir, "entities", "course.json")
        default_inter = os.path.join(self.base_dir, "MOOCCubeX.inter")
        self.inter_file = args.inter_file or (default_inter if os.path.exists(default_inter) else None)

        if args.bucket_count < 1:
            raise ValueError("bucket-count must be >= 1")

        os.makedirs(self.output_dir, exist_ok=True)

    def load_course_texts(self):
        print(f"1. Loading course metadata from {self.course_file} ...")
        courses = read_json_flexible(self.course_file)
        course_texts = {}
        for course in tqdm(courses, desc="Course metadata"):
            cid = course.get("id")
            if cid:
                course_texts[cid] = clean_course_text(course)
        print(f"   Loaded {len(course_texts)} courses with text.")
        return course_texts

    def _bucket_index(self, user_id, course_id):
        key = f"{user_id}\t{course_id}".encode("utf-8", errors="ignore")
        return zlib.crc32(key) % self.args.bucket_count

    def build_raw_buckets(self, tmp_dir, valid_course_ids):
        print("2. Streaming enrollments from MOOCCubeX/entities/user.json ...")
        bucket_paths = [os.path.join(tmp_dir, f"bucket_{idx:03d}.tsv") for idx in range(self.args.bucket_count)]
        bucket_files = [open(path, "w", encoding="utf-8") for path in bucket_paths]

        stats = {
            "users_scanned": 0,
            "users_with_valid_interactions": 0,
            "raw_events_written": 0,
            "invalid_time": 0,
            "invalid_course": 0,
            "mismatched_records": 0,
        }

        try:
            with open(self.user_file, "r", encoding="utf-8") as f:
                iterator = tqdm(f, desc="User enrollments", unit="user")
                for line in iterator:
                    if self.args.max_users is not None and stats["users_scanned"] >= self.args.max_users:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    stats["users_scanned"] += 1

                    try:
                        user_obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    user_id = user_obj.get("id")
                    course_order = user_obj.get("course_order", []) or []
                    enroll_time = user_obj.get("enroll_time", []) or []
                    if not user_id or not course_order:
                        continue
                    if len(course_order) != len(enroll_time):
                        stats["mismatched_records"] += 1
                        continue

                    user_written = 0
                    for raw_course_id, raw_time in zip(course_order, enroll_time):
                        course_id = normalize_course_id(raw_course_id)
                        if course_id not in valid_course_ids:
                            stats["invalid_course"] += 1
                            continue

                        timestamp = parse_enroll_time(raw_time)
                        if timestamp is None:
                            stats["invalid_time"] += 1
                            continue

                        dt_year = datetime.utcfromtimestamp(timestamp).year
                        if self.args.year_from is not None and dt_year < self.args.year_from:
                            continue
                        if self.args.year_to is not None and dt_year > self.args.year_to:
                            continue

                        bucket_idx = self._bucket_index(user_id, course_id)
                        bucket_files[bucket_idx].write(f"{user_id}\t{course_id}\t{timestamp}\t{raw_time}\n")
                        stats["raw_events_written"] += 1
                        user_written += 1

                    if user_written > 0:
                        stats["users_with_valid_interactions"] += 1
        finally:
            for fp in bucket_files:
                fp.close()

        print(
            "   Raw valid events written: "
            f"{stats['raw_events_written']} | users kept: {stats['users_with_valid_interactions']}"
        )
        return bucket_paths, stats

    def build_raw_buckets_from_inter(self, tmp_dir, valid_course_ids):
        print(f"2. Streaming enrollments from {self.inter_file} ...")
        bucket_paths = [os.path.join(tmp_dir, f"bucket_{idx:03d}.tsv") for idx in range(self.args.bucket_count)]
        bucket_files = [open(path, "w", encoding="utf-8") for path in bucket_paths]

        stats = {
            "rows_scanned": 0,
            "raw_events_written": 0,
            "invalid_time": 0,
            "invalid_course": 0,
            "bad_rows": 0,
            "source_mode": "inter_file",
        }

        try:
            with open(self.inter_file, "r", encoding="utf-8") as f:
                header = f.readline()
                if not header:
                    raise ValueError(f"Empty inter file: {self.inter_file}")

                iterator = tqdm(f, desc="Inter rows", unit="row")
                for line in iterator:
                    line = line.strip()
                    if not line:
                        continue

                    stats["rows_scanned"] += 1
                    parts = line.split("\t")
                    if len(parts) < 3:
                        stats["bad_rows"] += 1
                        continue

                    user_id, raw_course_id, ts_str = parts[0], parts[1], parts[2]
                    course_id = normalize_course_id(raw_course_id)
                    if course_id not in valid_course_ids:
                        stats["invalid_course"] += 1
                        continue

                    try:
                        timestamp = int(float(ts_str))
                    except ValueError:
                        stats["invalid_time"] += 1
                        continue

                    dt_year = datetime.utcfromtimestamp(timestamp).year
                    if self.args.year_from is not None and dt_year < self.args.year_from:
                        continue
                    if self.args.year_to is not None and dt_year > self.args.year_to:
                        continue

                    bucket_idx = self._bucket_index(user_id, course_id)
                    bucket_files[bucket_idx].write(f"{user_id}\t{course_id}\t{timestamp}\t{ts_str}\n")
                    stats["raw_events_written"] += 1
        finally:
            for fp in bucket_files:
                fp.close()

        print(f"   Raw valid events written: {stats['raw_events_written']}")
        return bucket_paths, stats

    def dedupe_and_sort_bucket(self, bucket_path, sorted_path):
        earliest = {}
        with open(bucket_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                user_id, course_id, ts_str, raw_time = line.split("\t", 3)
                timestamp = int(ts_str)
                key = (user_id, course_id)
                prev = earliest.get(key)
                if prev is None or timestamp < prev[0]:
                    earliest[key] = (timestamp, raw_time)

        rows = [
            (timestamp, user_id, course_id, raw_time)
            for (user_id, course_id), (timestamp, raw_time) in earliest.items()
        ]
        rows.sort(key=lambda x: (x[0], x[1], x[2]))

        with open(sorted_path, "w", encoding="utf-8") as f:
            for timestamp, user_id, course_id, raw_time in rows:
                f.write(f"{user_id}\t{course_id}\t{timestamp}\t{raw_time}\n")

        return len(rows)

    def dedupe_all_buckets(self, bucket_paths, tmp_dir):
        print("3. Deduplicating repeated user-course enrollments bucket by bucket ...")
        sorted_paths = []
        unique_rows = 0
        for idx, bucket_path in enumerate(tqdm(bucket_paths, desc="Buckets")):
            sorted_path = os.path.join(tmp_dir, f"bucket_{idx:03d}.sorted.tsv")
            sorted_paths.append(sorted_path)
            unique_rows += self.dedupe_and_sort_bucket(bucket_path, sorted_path)
        print(f"   Unique user-course interactions after dedupe: {unique_rows}")
        return sorted_paths, unique_rows

    def merge_sorted_buckets(self, sorted_paths, merged_path):
        print("4. Merging sorted buckets into one global time-ordered stream ...")
        handles = []
        heap = []
        total_rows = 0

        try:
            for file_idx, path in enumerate(sorted_paths):
                handle = open(path, "r", encoding="utf-8")
                handles.append(handle)
                line = handle.readline().rstrip("\n")
                if not line:
                    continue
                user_id, course_id, ts_str, raw_time = line.split("\t", 3)
                heapq.heappush(heap, (int(ts_str), user_id, course_id, raw_time, file_idx))

            with open(merged_path, "w", encoding="utf-8") as out_f:
                while heap:
                    timestamp, user_id, course_id, raw_time, file_idx = heapq.heappop(heap)
                    out_f.write(f"{user_id}\t{course_id}\t{timestamp}\t{raw_time}\n")
                    total_rows += 1

                    next_line = handles[file_idx].readline().rstrip("\n")
                    if next_line:
                        n_user_id, n_course_id, n_ts_str, n_raw_time = next_line.split("\t", 3)
                        heapq.heappush(
                            heap,
                            (int(n_ts_str), n_user_id, n_course_id, n_raw_time, file_idx),
                        )
        finally:
            for handle in handles:
                handle.close()

        print(f"   Merged rows: {total_rows}")
        return total_rows

    def build_dataframe(self, merged_path):
        print("5. Building final dataframe and aligned encodings ...")
        df = pd.read_csv(
            merged_path,
            sep="\t",
            names=["user_id", "course_id", "timestamp", "raw_time"],
            dtype={"user_id": str, "course_id": str, "timestamp": np.int64, "raw_time": str},
        )
        if df.empty:
            raise ValueError("No valid enrollments were found for MOOCCubeX.")

        if self.args.min_user_inter > 1:
            user_counts = df["user_id"].value_counts()
            keep_users = user_counts[user_counts >= self.args.min_user_inter].index
            df = df[df["user_id"].isin(keep_users)].copy()

        if self.args.min_item_inter > 1:
            item_counts = df["course_id"].value_counts()
            keep_items = item_counts[item_counts >= self.args.min_item_inter].index
            df = df[df["course_id"].isin(keep_items)].copy()

        if df.empty:
            raise ValueError("All interactions were filtered out. Lower min-user-inter or min-item-inter.")

        df = df.sort_values("timestamp").reset_index(drop=True)
        u_codes, user_uniques = pd.factorize(df["user_id"], sort=True)
        i_codes, item_uniques = pd.factorize(df["course_id"], sort=True)
        df["u_idx"] = u_codes.astype(np.int64)
        df["i_idx"] = i_codes.astype(np.int64)
        df["popularity"] = df.groupby("i_idx").cumcount().astype(np.int64)

        meta = {
            "n_users": int(len(user_uniques)),
            "n_items": int(len(item_uniques)),
            "source": "MOOCCubeX course enrollments",
        }
        return df, list(item_uniques), meta

    def extract_bert_features(self, texts):
        print(f"6. Extracting text embeddings with {self.args.bert_model} ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model)
            model = AutoModel.from_pretrained(self.args.bert_model)
        except Exception:
            fallback = "bert-base-uncased"
            print(f"   Failed to load {self.args.bert_model}, falling back to {fallback}.")
            tokenizer = AutoTokenizer.from_pretrained(fallback)
            model = AutoModel.from_pretrained(fallback)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        all_embs = []
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), self.args.batch_size), desc="BERT embeddings"):
                batch = texts[start:start + self.args.batch_size]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.args.max_length,
                    return_tensors="pt",
                ).to(device)
                outputs = model(**inputs)
                all_embs.append(outputs.last_hidden_state[:, 0, :].cpu())
        return torch.cat(all_embs, dim=0)

    def save_outputs(self, df, content_emb, meta, stats):
        stream_path = os.path.join(self.output_dir, "stream_data.pkl")
        content_path = os.path.join(self.output_dir, "content_emb.pt")
        meta_path = os.path.join(self.output_dir, "meta.json")
        stats_path = os.path.join(self.output_dir, "stats.json")

        df.to_pickle(stream_path, protocol=4)
        torch.save(content_emb, content_path)
        meta["content_dim"] = int(content_emb.shape[1])
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"7. Saved stream data to {stream_path}")
        print(f"   Saved content embeddings to {content_path}")
        print(f"   Saved metadata to {meta_path}")
        print(f"   Saved stats to {stats_path}")

    def process(self):
        course_texts = self.load_course_texts()
        valid_course_ids = set(course_texts.keys())

        with tempfile.TemporaryDirectory(prefix="mooccubex_course_", dir=self.output_dir) as tmp_dir:
            if self.inter_file and os.path.exists(self.inter_file):
                bucket_paths, bucket_stats = self.build_raw_buckets_from_inter(tmp_dir, valid_course_ids)
            else:
                bucket_paths, bucket_stats = self.build_raw_buckets(tmp_dir, valid_course_ids)
            sorted_paths, unique_rows = self.dedupe_all_buckets(bucket_paths, tmp_dir)
            merged_path = os.path.join(tmp_dir, "merged_sorted.tsv")
            merged_rows = self.merge_sorted_buckets(sorted_paths, merged_path)

            df, sorted_course_ids, meta = self.build_dataframe(merged_path)
            sorted_texts = [course_texts[cid] for cid in sorted_course_ids]
            content_emb = self.extract_bert_features(sorted_texts)

        stats = dict(bucket_stats)
        stats.update(
            {
                "unique_rows_after_dedupe": int(unique_rows),
                "rows_after_merge": int(merged_rows),
                "rows_after_filter": int(len(df)),
                "unique_users_after_filter": int(df["u_idx"].nunique()),
                "unique_items_after_filter": int(df["i_idx"].nunique()),
                "min_timestamp": int(df["timestamp"].min()),
                "max_timestamp": int(df["timestamp"].max()),
            }
        )
        self.save_outputs(df, content_emb, meta, stats)


def main():
    args = parse_args()
    processor = MOOCCubeXCourseProcessor(args)
    processor.process()


if __name__ == "__main__":
    main()
