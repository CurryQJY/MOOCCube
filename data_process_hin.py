"""
HIN-aware preprocessing for MOOCCube.

Outputs:
  - ./processed_data_hin/stream_data.pkl
  - ./processed_data_hin/content_emb.pt
  - ./processed_data_hin/meta.json

The important detail is stream alignment:
if ./processed_data/stream_data.pkl exists, this script reuses its
timestamp/u_idx/i_idx/popularity columns so HIN experiments stay on the same
stream as the base processed data.
"""

from collections import defaultdict
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_tsv_relation(filepath, reverse=False):
    mapping = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            src, dst = parts[0], parts[1]
            if reverse:
                mapping[dst].append(src)
            else:
                mapping[src].append(dst)
    return mapping


def load_entity_names(filepath):
    names = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            entity_id = data.get("id", "")
            name = data.get("name", "") or ""
            if entity_id and name:
                names[entity_id] = name
    return names


class HINDataProcessor:
    def __init__(
        self,
        base_dir="./MOOCCube",
        output_dir="./processed_data_hin",
        reference_stream_path="./processed_data/stream_data.pkl",
    ):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.reference_stream_path = reference_stream_path

        self.inter_file = os.path.join(base_dir, "relations", "user-course.json")
        self.course_file = os.path.join(base_dir, "entities", "course.json")
        self.school_course_file = os.path.join(base_dir, "relations", "school-course.json")
        self.teacher_course_file = os.path.join(base_dir, "relations", "teacher-course.json")
        self.course_concept_file = os.path.join(base_dir, "relations", "course-concept.json")
        self.teacher_entity_file = os.path.join(base_dir, "entities", "teacher.json")
        self.school_entity_file = os.path.join(base_dir, "entities", "school.json")

        os.makedirs(output_dir, exist_ok=True)
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _load_reference_stream(self, valid_course_ids):
        if not self.reference_stream_path or not os.path.exists(self.reference_stream_path):
            return None

        print(f"2. [HIN] Trying reference stream: {self.reference_stream_path}")
        ref_df = pd.read_pickle(self.reference_stream_path)
        required_cols = {"user_id", "course_id", "timestamp", "u_idx", "i_idx", "popularity"}
        missing = sorted(required_cols - set(ref_df.columns))
        if missing:
            raise ValueError(
                f"Reference stream is missing required columns {missing}: {self.reference_stream_path}"
            )

        before = len(ref_df)
        ref_df = ref_df[ref_df["course_id"].isin(valid_course_ids)].copy()
        print(f"   Reference stream filtered by valid courses: {before} -> {len(ref_df)}")
        if len(ref_df) == 0:
            raise ValueError("Reference stream is empty after filtering valid courses.")

        if "raw_time" not in ref_df.columns:
            ref_df["raw_time"] = ""

        ref_df = ref_df[
            ["user_id", "course_id", "raw_time", "timestamp", "u_idx", "i_idx", "popularity"]
        ].copy()
        ref_df = ref_df.sort_values("timestamp").reset_index(drop=True)

        course_order = (
            ref_df[["course_id", "i_idx"]]
            .drop_duplicates()
            .sort_values("i_idx")
            .reset_index(drop=True)
        )
        actual_i = course_order["i_idx"].astype(int).tolist()
        expected_i = list(range(len(course_order)))
        if actual_i != expected_i:
            raise ValueError("Reference stream i_idx is not contiguous from 0; cannot align HIN embeddings.")

        print(
            "   Reference stream alignment OK: "
            f"users={int(ref_df['u_idx'].max()) + 1}, items={len(course_order)}, rows={len(ref_df)}"
        )
        return ref_df, course_order["course_id"].tolist()

    def _load_graph_features(self):
        print("0. [HIN] Loading school-course relations...")
        school_course = load_tsv_relation(self.school_course_file)
        course_school = defaultdict(list)
        for school_id, courses in school_course.items():
            for course_id in courses:
                course_school[course_id].append(school_id)
        print(f"   School coverage: {len(course_school)} courses")

        print("0. [HIN] Loading teacher-course relations...")
        teacher_course = load_tsv_relation(self.teacher_course_file)
        course_teachers = defaultdict(list)
        for teacher_id, courses in teacher_course.items():
            for course_id in courses:
                course_teachers[course_id].append(teacher_id)
        print(f"   Teacher coverage: {len(course_teachers)} courses")

        print("0. [HIN] Loading course-concept relations...")
        course_concepts = load_tsv_relation(self.course_concept_file)
        print(f"   Concept coverage: {len(course_concepts)} courses")

        print("0. [HIN] Loading entity names...")
        teacher_names = load_entity_names(self.teacher_entity_file)
        school_names = load_entity_names(self.school_entity_file)
        print(f"   Teachers: {len(teacher_names)}, schools: {len(school_names)}")

        return course_school, course_teachers, course_concepts, teacher_names, school_names

    def _build_enriched_text(
        self,
        course_id,
        name,
        about,
        course_school,
        course_teachers,
        course_concepts,
        teacher_names,
        school_names,
    ):
        parts = []

        schools = course_school.get(course_id, [])
        if schools:
            school_labels = [school_names.get(school_id, school_id.replace("S_", "")) for school_id in schools]
            parts.append(f"[School] {'; '.join(school_labels)}")

        teachers = course_teachers.get(course_id, [])
        if teachers:
            teacher_labels = [teacher_names.get(teacher_id, teacher_id.replace("T_", "")) for teacher_id in teachers]
            parts.append(f"[Teacher] {'; '.join(teacher_labels[:5])}")

        concepts = course_concepts.get(course_id, [])
        if concepts:
            concept_labels = []
            for concept_id in concepts[:15]:
                concept_clean = concept_id.replace("K_", "")
                concept_parts = concept_clean.split("_")
                concept_labels.append(concept_parts[0] if concept_parts else concept_clean)
            parts.append(f"[Concept] {'; '.join(concept_labels)}")

        if name:
            parts.append(f"[Course] {name}")
        if about:
            about_clean = re.sub(r"<[^>]+>", "", about).strip()
            if about_clean:
                parts.append(f"[About] {about_clean}")

        return " ".join(parts) if parts else name or "Unknown Course"

    def _load_course_metadata_enriched(
        self,
        course_school,
        course_teachers,
        course_concepts,
        teacher_names,
        school_names,
    ):
        print(f"1. Loading enriched course metadata from {self.course_file} ...")
        course_texts = {}
        with open(self.course_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Course JSON"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                course_id = data.get("id")
                name = data.get("name", "") or ""
                about = data.get("about", "") or ""
                text = self._build_enriched_text(
                    course_id,
                    name,
                    about,
                    course_school,
                    course_teachers,
                    course_concepts,
                    teacher_names,
                    school_names,
                )
                if course_id and text:
                    course_texts[course_id] = text

        print(f"   Loaded {len(course_texts)} enriched course texts.")
        return course_texts

    def _extract_bert_features(self, texts, batch_size=32):
        print("5. Loading BERT model (bert-base-chinese)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            model = AutoModel.from_pretrained("bert-base-chinese")
        except Exception:
            print("   Falling back to bert-base-uncased...")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        all_embs = []
        print(f"   Extracting features on {device} with max_length=256 ...")
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="BERT Embedding"):
                batch = texts[start : start + batch_size]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)
                outputs = model(**inputs)
                all_embs.append(outputs.last_hidden_state[:, 0, :].cpu())
        return torch.cat(all_embs, dim=0)

    def _extract_time_from_course_id(self, course_id):
        course_id = str(course_id)
        match = re.search(r"(20[0-2]\d)", course_id)
        if match:
            year = int(match.group(1))
            base_time = (year - 1970) * 31536000
            return base_time + np.random.randint(0, 3600 * 24 * 30)
        return 1483228800 + np.random.randint(0, 31536000)

    def process(self):
        course_school, course_teachers, course_concepts, teacher_names, school_names = self._load_graph_features()
        course_texts = self._load_course_metadata_enriched(
            course_school,
            course_teachers,
            course_concepts,
            teacher_names,
            school_names,
        )
        valid_course_ids = set(course_texts.keys())

        ref_bundle = self._load_reference_stream(valid_course_ids)
        if ref_bundle is not None:
            df, sorted_cids = ref_bundle
            n_users = int(df["u_idx"].max()) + 1
            n_items = len(sorted_cids)
            print("3. [HIN] Reusing reference stream timestamp/u_idx/i_idx/popularity.")
        else:
            print(f"2. Loading interactions from {self.inter_file} ...")
            df = pd.read_csv(
                self.inter_file,
                sep="\t",
                header=None,
                names=["user_id", "course_id", "raw_time"],
                on_bad_lines="skip",
            )
            original_len = len(df)
            df = df[df["course_id"].isin(valid_course_ids)].copy()
            print(f"   Filter invalid courses: {original_len} -> {len(df)}")
            if len(df) == 0:
                raise ValueError("Interaction data has no valid course_id after filtering.")

            print("3. Building streaming timestamps...")
            tqdm.pandas(desc="Timestamp")
            df["timestamp"] = df["course_id"].progress_apply(self._extract_time_from_course_id)
            df = df.sort_values("timestamp").reset_index(drop=True)

            print("4. Encoding IDs...")
            df["u_idx"] = self.user_enc.fit_transform(df["user_id"])
            df["i_idx"] = self.item_enc.fit_transform(df["course_id"])
            n_users = len(self.user_enc.classes_)
            n_items = len(self.item_enc.classes_)
            sorted_cids = self.item_enc.inverse_transform(range(n_items))

            print("6. Computing popularity...")
            df["popularity"] = df.groupby("i_idx").cumcount()

        print("5. Building HIN-aware content embeddings...")
        sorted_texts = [course_texts[cid] for cid in sorted_cids]
        print("\n   === Enriched text examples ===")
        for idx in range(min(3, len(sorted_texts))):
            print(f"   [{idx}] {sorted_texts[idx][:200]}...")
        print("   =============================\n")

        content_emb = self._extract_bert_features(sorted_texts)
        print(f"   Embedding shape: {tuple(content_emb.shape)}")

        print("7. Saving outputs...")
        df.to_pickle(os.path.join(self.output_dir, "stream_data.pkl"))
        torch.save(content_emb, os.path.join(self.output_dir, "content_emb.pt"))

        meta = {
            "n_users": int(n_users),
            "n_items": int(n_items),
            "content_dim": int(content_emb.shape[1]),
        }
        with open(os.path.join(self.output_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        print(f"\n[Done] Saved HIN processed data to {self.output_dir}")
        print(f"   Users: {n_users}, items: {n_items}, content dim: {content_emb.shape[1]}")

        has_school = sum(1 for cid in sorted_cids if cid in course_school)
        has_teacher = sum(1 for cid in sorted_cids if cid in course_teachers)
        has_concept = sum(1 for cid in sorted_cids if cid in course_concepts)
        print(
            "   Graph coverage: "
            f"school {has_school}/{n_items} ({has_school / max(1, n_items) * 100:.1f}%), "
            f"teacher {has_teacher}/{n_items} ({has_teacher / max(1, n_items) * 100:.1f}%), "
            f"concept {has_concept}/{n_items} ({has_concept / max(1, n_items) * 100:.1f}%)"
        )


if __name__ == "__main__":
    processor = HINDataProcessor(
        base_dir="./MOOCCube",
        output_dir="./processed_data_hin",
    )
    processor.process()
