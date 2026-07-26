import argparse
import json
import os
import pickle
import re
import shutil
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_course_texts(course_file):
    course_file = Path(course_file)
    if not course_file.exists():
        raise FileNotFoundError(f"Course file not found: {course_file}")

    with course_file.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            rows = json.load(f)
        else:
            rows = [json.loads(line) for line in f if line.strip()]

    course_texts = {}
    for row in rows:
        course_id = row.get("id")
        if not course_id:
            continue
        name = row.get("name", "") or ""
        about = row.get("about", "") or row.get("intro", "") or ""
        about = re.sub(r"<[^>]+>", " ", about).replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", f"{name}. {about}").strip()
        if text:
            course_texts[course_id] = text[:1200]
    return course_texts


def load_scores(path, fresh=False):
    path = Path(path)
    if fresh or not path.exists():
        return {}
    with path.open("rb") as f:
        scores = pickle.load(f)
    return {
        (int(k[0]), int(k[1])): float(v)
        for k, v in scores.items()
        if isinstance(k, tuple) and len(k) == 2
    }


def save_scores(path, scores):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(scores, f)
    tmp_path.replace(path)


def build_prompt(history_texts, target_text):
    history = "\n".join(f"{idx}. {text}" for idx, text in enumerate(history_texts, start=1))
    return (
        "请根据学生最近选过的课程，判断他/她选择目标课程的可能性。\n\n"
        "评分标准：\n"
        "0.0-0.2：几乎无关，主题、能力层次或学习路径差异很大。\n"
        "0.3-0.4：弱相关，只有少量背景知识或泛化主题相连。\n"
        "0.5-0.6：中等相关，主题、技能或先修知识有明显衔接。\n"
        "0.7-0.8：强相关，目标课程很像历史课程的后续、扩展或同方向进阶。\n"
        "0.9-1.0：高度匹配，几乎是自然的下一门课或同一学习路径中的核心课程。\n\n"
        "判断时优先考虑：课程主题连续性、先修/后续关系、技能层次递进、学科方向一致性；"
        "不要只因为课程都属于大学课程就给高分。\n\n"
        f"学生最近课程：\n{history}\n\n"
        f"目标课程：\n{target_text}\n\n"
        "只输出一个 0 到 1 之间的小数，最多两位小数，不要解释。"
    )


def parse_score(text):
    match = re.search(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", text)
    if not match:
        return 0.5
    return max(0.0, min(1.0, float(match.group(0))))


def collect_tasks(df, course_texts, scores, threshold, max_history):
    df = df.sort_values("timestamp").reset_index(drop=True)
    user_hist = {}
    tasks = []

    for row in df[["user_id", "course_id", "u_idx", "i_idx", "popularity"]].itertuples(index=False):
        user_id = row.user_id
        course_id = row.course_id
        pair = (int(row.u_idx), int(row.i_idx))

        hist = user_hist.setdefault(user_id, [])
        if int(row.popularity) < threshold and hist and pair not in scores:
            history_ids = hist[-max_history:]
            history_texts = [course_texts[cid] for cid in history_ids if cid in course_texts]
            target_text = course_texts.get(course_id)
            if history_texts and target_text:
                tasks.append((pair, history_texts, target_text))

        hist.append(course_id)

    return tasks


def load_model(model_id, use_4bit=True, local_files_only=True):
    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as exc:
            print(f"Warning: 4-bit config unavailable, using default precision: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def score_batch(model, tokenizer, prompts, max_new_tokens):
    system_prompt = (
        "你是严格的慕课选课推荐评分器。你的任务是输出校准后的概率分数。"
        "必须只输出一个 0 到 1 之间的小数，不输出任何解释、标点或单位。"
    )
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]
    texts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    return [parse_score(text) for text in decoded]


def main():
    parser = argparse.ArgumentParser(description="Regenerate index-clean LLM scores for HIN data.")
    parser.add_argument("--data-dir", default="processed_data_hin_clean")
    parser.add_argument("--course-file", default="MOOCCube/entities/course.json")
    parser.add_argument("--model-id", default=os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--max-history", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing scores and rebuild from empty.")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    stream_path = data_dir / "stream_data.pkl"
    if not stream_path.exists():
        raise FileNotFoundError(f"Missing stream data: {stream_path}")

    output_path = Path(args.output) if args.output else data_dir / "llm_scores.pkl"
    df = pd.read_pickle(stream_path)
    course_texts = load_course_texts(args.course_file)
    scores = load_scores(output_path, fresh=args.fresh)

    tasks = collect_tasks(
        df=df,
        course_texts=course_texts,
        scores=scores,
        threshold=args.threshold,
        max_history=args.max_history,
    )
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"Data dir: {data_dir}")
    print(f"Output: {output_path}")
    print(f"Existing scores: {len(scores)}")
    print(f"Pending tasks: {len(tasks)}")
    print(f"Threshold: popularity < {args.threshold}")

    if args.dry_run:
        return
    if not tasks:
        save_scores(output_path, scores)
        print("No pending tasks; score file is unchanged.")
        return

    if output_path.exists() and not args.no_backup:
        backup_path = output_path.with_name(output_path.name + ".bak_before_rescore")
        if not backup_path.exists():
            shutil.copy2(output_path, backup_path)
            print(f"Backup written: {backup_path}")

    model, tokenizer = load_model(
        args.model_id,
        use_4bit=not args.no_4bit,
        local_files_only=not args.allow_download,
    )

    new_count = 0
    for start in tqdm(range(0, len(tasks), args.batch_size), desc="LLM scoring"):
        batch = tasks[start : start + args.batch_size]
        prompts = [build_prompt(history_texts, target_text) for _, history_texts, target_text in batch]
        batch_scores = score_batch(model, tokenizer, prompts, args.max_new_tokens)
        for (pair, _, _), score in zip(batch, batch_scores):
            scores[pair] = score
            new_count += 1
        if new_count % args.save_every == 0:
            save_scores(output_path, scores)

    save_scores(output_path, scores)
    print(f"New scores: {new_count}")
    print(f"Total scores: {len(scores)}")


if __name__ == "__main__":
    main()
