import pandas as pd
import json
import os
import pickle
from collections import defaultdict
from tqdm import tqdm

base_dir = "./MOOCCube"
rel_dir = os.path.join(base_dir, "relations")
out_dir = "./processed_data_domain"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("1. Loading prerequisite-dependency.json...")
prerequisites = defaultdict(list)
with open(os.path.join(rel_dir, "prerequisite-dependency.json"), "r", encoding="utf-8") as f:
    for line in f:
        src, dst = line.strip().split("\t")
        # src is prerequisite for dst
        prerequisites[dst].append(src)

print("2. Loading video-concept.json...")
video_concepts = defaultdict(set)
with open(os.path.join(rel_dir, "video-concept.json"), "r", encoding="utf-8") as f:
    for line in f:
        vid, concept = line.strip().split("\t")
        video_concepts[vid].add(concept)

print("3. Loading course-video.json...")
course_videos = defaultdict(list)
with open(os.path.join(rel_dir, "course-video.json"), "r", encoding="utf-8") as f:
    for line in f:
        cid, vid = line.strip().split("\t")
        course_videos[cid].append(vid)

print("4. Mapping Course -> Concepts and identifying Required Concepts...")
course_concepts = defaultdict(set)
all_concepts = set()
for cid, vids in course_videos.items():
    for vid in vids:
        concepts = video_concepts.get(vid, set())
        course_concepts[cid].update(concepts)
        all_concepts.update(concepts)

concept_list = sorted(list(all_concepts))
concept2id = {c: i for i, c in enumerate(concept_list)}
print(f"   Total Unique Concepts: {len(concept_list)}")

course_req_concepts = defaultdict(set)
for cid, concepts in course_concepts.items():
    req_set = set()
    for c in concepts:
        if c in prerequisites:
            req_set.update(prerequisites[c])
    # Optionally: Concepts covered in the course might also be prerequisites for other concepts in the same course, 
    # but we usually care about what the user needs BEFORE starting the course.
    # Exclude concepts that are already taught in the course.
    req_set = req_set - concepts
    course_req_concepts[cid] = req_set

print("5. Loading user-video.json to build user mastery...")
# Simulate user mastery based on all videos watched.
user_mastery = defaultdict(set)
try:
    with open(os.path.join(rel_dir, "user-video.json"), "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing user-video"):
            try:
                uid, vid = line.strip().split("\t")
                if vid in video_concepts:
                    user_mastery[uid].update(video_concepts[vid])
            except Exception:
                continue
except FileNotFoundError:
    print("Warning: user-video.json not found. Using empty user mastery.")


print("6. Aligning IDs with stream_data.pkl...")
stream_data_path = "./processed_data_hin/stream_data.pkl"
df = pd.read_pickle(stream_data_path)

u_map = df.drop_duplicates(subset=['user_id', 'u_idx'])[['user_id', 'u_idx']].set_index('user_id')['u_idx'].to_dict()
i_map = df.drop_duplicates(subset=['course_id', 'i_idx'])[['course_id', 'i_idx']].set_index('course_id')['i_idx'].to_dict()

def to_concept_ids(concept_set):
    return [concept2id[c] for c in concept_set if c in concept2id]

user_mastery_idx = {}
for uid, uidx in u_map.items():
    if uid in user_mastery:
        user_mastery_idx[uidx] = to_concept_ids(user_mastery[uid])
    else:
        user_mastery_idx[uidx] = []

course_concepts_idx = {}
course_req_concepts_idx = {}
for cid, iidx in i_map.items():
    if cid in course_concepts:
        course_concepts_idx[iidx] = to_concept_ids(course_concepts[cid])
        course_req_concepts_idx[iidx] = to_concept_ids(course_req_concepts[cid])
    else:
        course_concepts_idx[iidx] = []
        course_req_concepts_idx[iidx] = []

# Validate the generated data
avg_course_concepts = sum(len(v) for v in course_concepts_idx.values()) / max(len(course_concepts_idx), 1)
avg_req_concepts = sum(len(v) for v in course_req_concepts_idx.values()) / max(len(course_req_concepts_idx), 1)
avg_user_mastery = sum(len(v) for v in user_mastery_idx.values()) / max(len(user_mastery_idx), 1)

print(f"   Avg concepts per course: {avg_course_concepts:.2f}")
print(f"   Avg req concepts per course: {avg_req_concepts:.2f}")
print(f"   Avg mastery concepts per user: {avg_user_mastery:.2f}")

print("7. Saving domain features...")
with open(os.path.join(out_dir, "user_mastery.pkl"), "wb") as f:
    pickle.dump(user_mastery_idx, f)
with open(os.path.join(out_dir, "course_concepts.pkl"), "wb") as f:
    pickle.dump(course_concepts_idx, f)
with open(os.path.join(out_dir, "course_req_concepts.pkl"), "wb") as f:
    pickle.dump(course_req_concepts_idx, f)

meta = {
    'n_concepts': len(concept_list),
    'concept2id': concept2id
}
with open(os.path.join(out_dir, "concept_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)

print("Done generating domain specific features.")
