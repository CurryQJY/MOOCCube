import argparse
import json
import os
import shutil
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from transformers import AutoTokenizer

DEFAULT_MAX_WORKERS = 64
DEFAULT_RPS = 8
DEFAULT_WARMUP = 120

API_TIMEOUT = 60
API_KEY = os.getenv("QWEN_API_KEY", "EMPTY")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-235b-a22b-instruct-2507"

DEFAULT_TOKENIZER_PATH = "qwen3_pretrain/pretrained/Qwen3-32B"
DEFAULT_INPUT_PATH = "data/{dataset}/resource_info.json"
DEFAULT_OUTPUT_PATH = "data/{dataset}/resource_info_summary.json"

MAX_CONTEXT_TOKENS = 32768
REPLY_RESERVE_TOKENS = 200
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - REPLY_RESERVE_TOKENS

thread_local = threading.local()
global_tokenizer = None


class GradualRateLimiter:
    def __init__(self, target_rps, warmup_seconds=60):
        self.target_rps = target_rps
        self.warmup_seconds = warmup_seconds
        self.initial_rps = max(0.5, target_rps * 0.1)
        self.lock = threading.Lock()
        self.start_time = None
        self.next_available_time = 0

    def acquire(self):
        with self.lock:
            now = time.time()
            if self.start_time is None:
                self.start_time = now
                self.next_available_time = now

            elapsed = now - self.start_time
            if elapsed >= self.warmup_seconds:
                current_rps = self.target_rps
            else:
                progress = elapsed / self.warmup_seconds
                current_rps = self.initial_rps + (self.target_rps - self.initial_rps) * progress

            if self.next_available_time < now:
                self.next_available_time = now

            interval = 1.0 / current_rps
            my_slot = self.next_available_time
            self.next_available_time += interval

            wait_time = my_slot - now

        if wait_time > 0:
            time.sleep(wait_time)
        return wait_time


_global_rate_limiter = None
_rate_limiter_lock = threading.Lock()


def set_rate_limiter(rps, warmup):
    global _global_rate_limiter
    with _rate_limiter_lock:
        if rps and rps > 0:
            _global_rate_limiter = GradualRateLimiter(rps, warmup)


def get_rate_limiter():
    return _global_rate_limiter


def get_openai_client():
    if not hasattr(thread_local, "openai_client"):
        thread_local.openai_client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE,
            timeout=API_TIMEOUT,
        )
    return thread_local.openai_client


def load_status_file(status_path):
    path_obj = Path(status_path)
    if path_obj.exists():
        try:
            content = path_obj.read_text(encoding="utf-8").strip()
            return json.loads(content) if content else {}
        except Exception:
            return {}
    return {}


def save_status_file(status_path, status_dict):
    try:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Status save failed: {e}")


class OrderedWriter:
    def __init__(self, output_path, status_path, status_dict):
        self.output_path = output_path
        self.status_path = status_path
        self.status_dict = status_dict
        self.lock = threading.Lock()
        self.buffer = {}
        self.next_write_idx = 0
        self.file_handle = None

    def open(self):
        self.file_handle = open(self.output_path, "w", encoding="utf-8")

    def write(self, idx, item, success):
        with self.lock:
            self.buffer[idx] = (item, success)

            while self.next_write_idx in self.buffer:
                item_to_write, success_flag = self.buffer.pop(self.next_write_idx)

                if success_flag and item_to_write:
                    self.file_handle.write(json.dumps(item_to_write, ensure_ascii=False) + "\n")

                self.status_dict[str(self.next_write_idx)] = "success" if success_flag else "failed"

                self.next_write_idx += 1

                if len(self.buffer) > 100:
                    print(
                        f"\r[Warning] Writer is waiting for ID: {self.next_write_idx}, "
                        f"but {len(self.buffer)} subsequent results are already queued.",
                        end="",
                        flush=True,
                    )

                if self.next_write_idx % 20 == 0:
                    self.file_handle.flush()
                    save_status_file(self.status_path, self.status_dict)

    def close(self):
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            save_status_file(self.status_path, self.status_dict)


def construct_smart_prompt(item: dict, max_tokens: int) -> str:
    prompt_prefix = (
        "Assume you are an educational content analysis expert. Below is the basic information of a specific resource.\n"
        f"Resource Type: {item.get('resource_type', 'Unknown')}\n"
        f"Resource Title: {item.get('resource_title', 'Unknown')}\n"
        f"Resource Content: {item.get('resource_content', 'Unknown')}\n"
        f"Belong Course: {item.get('related_course', None)}\n"
        f"Related Concepts: {item.get('related_concepts', None)}\n"
    )

    prompt_suffix = (
        "Please provide a concise summary that describe the core knowledge of the above resource, "
        "the summary should be no more than 80 words.\n"
    )

    prefix_tokens = global_tokenizer.encode(prompt_prefix, add_special_tokens=False)
    suffix_tokens = global_tokenizer.encode(prompt_suffix, add_special_tokens=False)

    fixed_cost = len(prefix_tokens) + len(suffix_tokens)
    content_budget = max_tokens - fixed_cost - 20

    if content_budget <= 0:
        return prompt_prefix + "[Content Removed]" + prompt_suffix

    raw_content = item.get("resource_content", "")
    content_tokens = global_tokenizer.encode(raw_content, add_special_tokens=False)

    if len(content_tokens) > content_budget:
        truncated_ids = content_tokens[:content_budget]
        final_content = global_tokenizer.decode(truncated_ids)
    else:
        final_content = raw_content

    return prompt_prefix + final_content + prompt_suffix


def process_single_item_task(index, line_content):
    result_item = None
    try:
        limiter = get_rate_limiter()
        if limiter:
            limiter.acquire()

        item = json.loads(line_content)
        resource_id = item.get("resource_id")

        final_prompt = construct_smart_prompt(item, MAX_INPUT_TOKENS)

        client = get_openai_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=REPLY_RESERVE_TOKENS,
            temperature=0.1,
            top_p=0.9,
        )

        if not response.choices or not response.choices[0].message:
            print("index:", index, "error response:", response)
            return index, None, False, response

        summary_text = response.choices[0].message.content

        if summary_text:
            result_item = {"id": resource_id, "summary": summary_text.strip()}
            return index, result_item, True, None

        return index, None, False, "Empty Response"

    except Exception as e:
        print(f"Error processing index {index}: {e}\n{traceback.format_exc()}")
        return index, None, False, str(e)


def run_concurrent_processing(input_path, output_path, max_workers, rps, warmup, resume):
    from tqdm import tqdm

    input_path = Path(input_path)
    output_path = Path(output_path)
    status_path = output_path.with_suffix(f"{output_path.suffix}.status")

    rewrite_path = output_path.with_suffix(".processing.jsonl")

    print(f"\n{'=' * 60}")
    print(f"Concurrent processing started: Workers={max_workers}, RPS={rps}")
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"{'=' * 60}")

    print("1. load origin files...")
    with open(input_path, "r", encoding="utf-8") as f:
        source_lines = [line.strip() for line in f if line.strip()]
    total_lines = len(source_lines)
    print(f"   ✓ the number of origin data: {total_lines} ")

    status_dict = load_status_file(status_path)

    existing_results_map = {}
    if resume and output_path.exists():
        print("2. load previous results for resuming execution...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        existing_results_map[str(obj["id"])] = obj
                except Exception:
                    pass
        print(f"   ✓ cached {len(existing_results_map)} completed records")

    writer = OrderedWriter(rewrite_path, status_path, status_dict)
    writer.open()

    tasks = []
    pbar = tqdm(total=total_lines, desc="Processing")
    stats = {"success": 0, "failed": 0, "skipped": 0}
    error_counter = Counter()

    def simplify_error(err_msg):
        e = str(err_msg).lower()
        if "rate limit" in e:
            return "RateLimit"
        if "timeout" in e:
            return "Timeout"
        if "connection" in e:
            return "ConnectionErr"
        return str(err_msg)[:20]

    def handle_result(idx, res_item, is_success, err_msg):
        writer.write(idx, res_item, is_success)

        if is_success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            error_counter[simplify_error(err_msg)] += 1

        top_err = error_counter.most_common(1)
        err_str = f"{top_err[0][0]}:{top_err[0][1]}" if top_err else ""
        pbar.set_postfix_str(
            f"OK:{stats['success']} Skip:{stats['skipped']} Fail:{stats['failed']} {err_str}"
        )
        pbar.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(total_lines):
            line_content = source_lines[i]
            idx_str = str(i)

            should_skip = False
            cached_item = None

            if resume and status_dict.get(idx_str) == "success":
                try:
                    tmp_json = json.loads(line_content)
                    tmp_id = str(tmp_json.get("resource_id"))
                    if tmp_id in existing_results_map:
                        cached_item = existing_results_map[tmp_id]
                        should_skip = True
                except Exception:
                    pass

            if should_skip:
                stats["skipped"] += 1
                handle_result(i, cached_item, True, None)
            else:
                future = executor.submit(process_single_item_task, i, line_content)
                tasks.append(future)

            while len(tasks) > max_workers * 2:
                kept_tasks = []
                for task in tasks:
                    if task.done():
                        try:
                            r_idx, r_item, r_suc, r_err = task.result()
                            handle_result(r_idx, r_item, r_suc, r_err)
                        except Exception as e:
                            print(f"CRITICAL ERROR retrieving task result: {e}")
                    else:
                        kept_tasks.append(task)

                tasks = kept_tasks

                if len(tasks) > max_workers * 2:
                    time.sleep(0.1)

        for ft in as_completed(tasks):
            r_idx, r_item, r_suc, r_err = ft.result()
            handle_result(r_idx, r_item, r_suc, r_err)

    pbar.close()
    writer.close()

    print("\nTask is completed...")
    if output_path.exists():
        backup_path = output_path.with_suffix(".bak.jsonl")
        shutil.move(str(output_path), str(backup_path))
        print(f"   ✓ Old files are backuped to: {backup_path}")

    shutil.move(str(rewrite_path), str(output_path))
    print(f"   ✓ results saved to: {output_path}")
    print(f"Statics: Success {stats['success']} (skipped {stats['skipped']}), failed {stats['failed']}")


def main():
    parser = argparse.ArgumentParser(description="Qwen Summary Generator (High Concurrency)")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--rps", type=float, default=DEFAULT_RPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if args.rps:
        set_rate_limiter(args.rps, args.warmup)

    global global_tokenizer
    print(f"Loading Tokenizer: {args.tokenizer_path}")
    try:
        global_tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    run_concurrent_processing(
        input_path=args.input,
        output_path=args.output,
        max_workers=args.workers,
        rps=args.rps,
        warmup=args.warmup,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
