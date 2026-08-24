import argparse
import ast
import json
import os
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

MAX_WORKERS = 64
API_TIMEOUT = 60
API_KEY = os.getenv("QWEN_API_KEY", "EMPTY")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MAX_RETRIES = 3
RETRY_DELAY = 2

DEFAULT_MODEL_NAME = "qwen3-235b-a22b-instruct-2507"

thread_local = threading.local()


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


def build_segment_prompt(behavior_records):
    prompt_lines = [
        "Please analyze the following learner behavior sequence, and summarize the characteristics of this learning stage.\n",
        "Behavior record format: [Resource ID, Title, Behavior Type, Timestamps, Engagement, Related Course, Related Knowledge Concepts]\n",
    ]

    for i, record in enumerate(behavior_records, 1):
        try:
            resource_id = record[0]
            title = record[1]
            behavior_type = record[2]
            timestamp = record[3]
            engagement = record[4]
            course = record[5]
            concepts = record[6] if len(record) > 6 else "N/A"

            prompt_lines.append(
                f"{i}. [{resource_id}] {title} | {behavior_type} | {timestamp} | "
                f"Engagement:{engagement} | {course} | Concepts:{concepts}"
            )
        except Exception as e:
            prompt_lines.append(f"{i}. [error: {str(e)}]")

    prompt_lines.append("\nPlease summarize the characteristics of this learning stage in 2-3 sentences, including the learning focus, behavioral patterns, and time span.")

    return "\n".join(prompt_lines)


def get_model_response_streaming(
    prompt_text,
    enable_thinking=True,
    max_tokens=32768,
    temperature=0.7,
    retry_count=0,
):
    simulated_response = {
        "id": None,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": None,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "", "reasoning_content": ""},
                "finish_reason": None,
            }
        ],
        "usage": None,
    }
    response_info = {
        "raw_response": None,
        "finish_reason": None,
        "chunk_count": 0,
        "error": None,
        "rps_wait_time": 0,
    }

    try:
        limiter = get_rate_limiter()
        if limiter:
            wait_time = limiter.acquire()
            response_info["rps_wait_time"] = wait_time

        client = get_openai_client()
        messages = [{"role": "user", "content": prompt_text}]

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}} if enable_thinking else None,
        )

        full_content = []
        full_reasoning = []
        chunk_count = 0

        for chunk in stream:
            chunk_count += 1
            if hasattr(chunk, "data") and chunk.data == "request_error":
                raise Exception(getattr(chunk, "message", "Unknown API Error"))

            if simulated_response["id"] is None:
                simulated_response["id"] = getattr(chunk, "id", None)
                simulated_response["model"] = getattr(chunk, "model", None)

            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason:
                    simulated_response["choices"][0]["finish_reason"] = choice.finish_reason
                    response_info["finish_reason"] = choice.finish_reason

                if hasattr(delta, "content") and delta.content:
                    full_content.append(delta.content)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    full_reasoning.append(delta.reasoning_content)

            if hasattr(chunk, "usage") and chunk.usage:
                simulated_response["usage"] = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

        final_content_str = "".join(full_content)
        final_reasoning_str = "".join(full_reasoning)

        simulated_response["choices"][0]["message"]["content"] = final_content_str
        simulated_response["choices"][0]["message"]["reasoning_content"] = final_reasoning_str
        response_info["chunk_count"] = chunk_count
        response_info["raw_response"] = simulated_response

        if not final_content_str and not final_reasoning_str:
            response_info["error"] = (
                f"empty_response_from_stream (finish_reason: {response_info.get('finish_reason')})"
            )
            return None, response_info

        final_text = final_content_str
        if final_reasoning_str:
            final_text = f"<think>\n{final_reasoning_str}</think>\n\n{final_text}"

        return final_text, response_info

    except Exception as e:
        error_msg = str(e)
        response_info["error"] = error_msg
        response_info["raw_response"] = {"error": error_msg, "traceback": traceback.format_exc()}

        should_retry = any(
            [
                "connection" in error_msg.lower(),
                "timeout" in error_msg.lower(),
                "ssl" in error_msg.lower(),
                "remote host" in error_msg,
                "connecting" in error_msg,
                "timed out" in error_msg.lower(),
                "reset" in error_msg.lower(),
            ]
        )

        if should_retry and retry_count < MAX_RETRIES:
            delay = RETRY_DELAY * (2**retry_count)
            time.sleep(delay)
            return get_model_response_streaming(
                prompt_text,
                enable_thinking,
                max_tokens,
                temperature,
                retry_count + 1,
            )

        return None, response_info


def process_single_segment(user_id, segment_id, segment_data_str):
    try:
        behavior_records = ast.literal_eval(segment_data_str)

        if not behavior_records:
            return user_id, segment_id, None, False, "empty_segment"

        prompt_text = build_segment_prompt(behavior_records)

        response_text, response_info = get_model_response_streaming(prompt_text)

        if response_text:
            return user_id, segment_id, response_text, True, None

        error_msg = response_info.get("error", "unknown_error")
        return user_id, segment_id, None, False, error_msg

    except Exception as e:
        return user_id, segment_id, None, False, f"process_error: {str(e)}"


class JSONLWriter:
    def __init__(self, output_path, status_path, status_dict):
        self.output_path = output_path
        self.status_path = status_path
        self.status_dict = status_dict
        self.lock = threading.Lock()
        self.file_handle = None
        self.write_count = 0

    def open(self):
        self.file_handle = open(self.output_path, "a", encoding="utf-8")

    def write(self, user_id, segment_id, summary, success):
        with self.lock:
            status_key = f"{user_id}::{segment_id}"

            if success and summary:
                record = {
                    "user_id": str(user_id),
                    "segment_id": str(segment_id),
                    "summary": summary,
                }
                self.file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.status_dict[status_key] = "success"
            else:
                self.status_dict[status_key] = "failed"

            self.write_count += 1

            if self.write_count % 20 == 0:
                self.file_handle.flush()
                save_status_file(self.status_path, self.status_dict)

    def close(self):
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            save_status_file(self.status_path, self.status_dict)


def convert_jsonl_to_nested_json(jsonl_path, output_path):
    result = {}

    if not Path(jsonl_path).exists():
        print(f"Intermediate file not found: {jsonl_path}")
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                uid = str(item["user_id"])
                sid = str(item["segment_id"])
                summary = item["summary"]

                if uid not in result:
                    result[uid] = {}
                result[uid][sid] = summary
            except Exception as e:
                print(f"Failed to parse line: {e}")

    sorted_result = dict(
        sorted(result.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_result, f, ensure_ascii=False, indent=2)

    print(f"Converted to nested JSON: {output_path}")
    print(f"Users: {len(sorted_result)}")
    total_segments = sum(len(segs) for segs in sorted_result.values())
    print(f"Total segments: {total_segments}")


def batch_process_segments(input_path, output_path, limit_users, max_workers, rps, warmup):
    input_path = Path(input_path)
    output_path = Path(output_path)

    jsonl_path = output_path.with_suffix(".jsonl")
    status_path = output_path.with_suffix(".status")

    print(f"\n{'=' * 60}")
    print("\033[1m Batch summarization script for user behavior segments\033[0m")
    print(f"  max workers: {max_workers}")
    print(f"  RPS: {'no limit' if not rps else rps}")
    print(f"  warmup: {warmup}s")
    print(f"  model: \033[31m{MODEL_NAME}\033[0m")
    print(f"{'=' * 60}")
    print(f"input file: {input_path}")
    print(f"middle file: {jsonl_path}")
    print(f"output file: {output_path}")

    print("\n[1/5] Load input file...")
    with open(input_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    total_users = len(source_data)
    print(f"  ✓ Total user number: {total_users}")

    total_tasks = 0
    for _, segments in source_data.items():
        total_tasks += len(segments)
    print(f"  ✓ Total task number: {total_tasks} (all segments in all users)")

    print("\n[2/5] Scan existing results...")
    completed_map = {}
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    uid = str(item["user_id"])
                    sid = str(item["segment_id"])
                    completed_map[f"{uid}::{sid}"] = True
                except Exception:
                    pass
    print(f"  ✓ Completed tasks: {len(completed_map)}")

    print("\n[3/5] Load status file...")
    status_dict = load_status_file(status_path)
    print(f"  ✓ Status records: {len(status_dict)}")

    print("\n[4/5] Build task queue...")
    task_queue = []

    user_limit = min(limit_users, total_users) if limit_users else total_users
    processed_users = 0

    for user_id in sorted(source_data.keys(), key=lambda x: int(x) if x.isdigit() else x):
        if processed_users >= user_limit:
            break
        processed_users += 1

        segments = source_data[user_id]
        for segment_id, segment_data_str in segments.items():
            status_key = f"{user_id}::{segment_id}"

            if status_dict.get(status_key) == "success" and status_key in completed_map:
                continue

            task_queue.append((user_id, segment_id, segment_data_str))

    print(f"  ✓ Pending tasks: {len(task_queue)}")

    if len(task_queue) == 0:
        print("\nAll tasks completed, converting format...")
        convert_jsonl_to_nested_json(jsonl_path, output_path)
        return

    print(f"\n[5/5] Start processing ({len(task_queue)} tasks)...\n")

    writer = JSONLWriter(jsonl_path, status_path, status_dict)
    writer.open()

    stats = {"success": 0, "failed": 0, "cached": len(completed_map)}
    error_counter = Counter()

    pbar = tqdm(total=len(task_queue), desc="Processing")

    def simplify_error(err_msg):
        e = str(err_msg).lower()
        if "scale requests" in e:
            return "RateLimit_Smooth"
        if "exceeded" in e:
            return "RateLimit_Quota"
        if "connecting" in err_msg or "remote host" in err_msg:
            return "Conn_Timeout_CN"
        if "ssl" in e:
            return "SSL_Error"
        if "timeout" in e or "timed out" in e:
            return "Timeout"
        if "transport connection" in e:
            return "Conn_Transport"
        if "connection reset" in e or "reset" in e:
            return "Conn_Reset"
        if "connection error" in e:
            return "Conn_Error"
        if "connection" in e:
            return "Conn_Other"
        if "copying content" in e:
            return "Stream_Error"
        if "incomplete" in e:
            return "Incomplete_Read"
        if "inappropriate" in e:
            return "Safety_Block"
        if "empty_response" in e:
            return "Empty_Response"
        if "empty_segment" in e:
            return "Empty_Segment"
        return err_msg[:25].strip()

    def handle_result(user_id, segment_id, summary, success, error):
        writer.write(user_id, segment_id, summary, success)

        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
            simple_err = simplify_error(error or "unknown")
            error_counter[simple_err] += 1

        top_errors = error_counter.most_common(3)
        if top_errors:
            err_str = ", ".join([f"{k}({v})" for k, v in top_errors])
            postfix_str = (
                f"✓{stats['success']} ✗{stats['failed']} 📦{stats['cached']} | Err: {err_str}"
            )
        else:
            postfix_str = f"✓{stats['success']} ✗{stats['failed']} 📦{stats['cached']}"
        pbar.set_postfix_str(postfix_str)
        pbar.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for user_id, segment_id, segment_data_str in task_queue:
            future = executor.submit(process_single_segment, user_id, segment_id, segment_data_str)
            futures.append(future)

        for future in as_completed(futures):
            user_id, segment_id, summary, success, error = future.result()
            handle_result(user_id, segment_id, summary, success, error)

    pbar.close()
    writer.close()

    if stats["failed"] > 0:
        print("\n" + "=" * 30)
        print("Error summary report (Top 10):")
        for err, count in error_counter.most_common(10):
            print(f"   - {err}: {count} ")
        print("=" * 30 + "\n")

    print("\nCompleted")
    print(f"   success: {stats['success']}")
    print(f"   failed: {stats['failed']}")
    print(f"   cached: {stats['cached']}")

    print("\nPerform format conversion...")
    convert_jsonl_to_nested_json(jsonl_path, output_path)

    print(f"\nAll completed. Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input path")
    parser.add_argument("--output", type=str, required=True, help="output path")
    parser.add_argument("--limit-users", type=int, default=None, help="limit users")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="workers")
    parser.add_argument("--rps", type=float, default=16)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)

    args = parser.parse_args()

    global MODEL_NAME
    MODEL_NAME = args.model_name

    if args.rps:
        set_rate_limiter(args.rps, args.warmup)

    batch_process_segments(
        input_path=args.input,
        output_path=args.output,
        limit_users=args.limit_users,
        max_workers=args.workers,
        rps=args.rps,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
