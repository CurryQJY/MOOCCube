"""Examine exact file formats of MOOCCube relation and entity files."""
import os

files = [
    ("relations/school-course.json", "school-course"),
    ("relations/teacher-course.json", "teacher-course"),
    ("relations/course-concept.json", "course-concept"),
    ("relations/course-video.json",   "course-video"),
    ("relations/prerequisite-dependency.json", "prerequisite"),
    ("entities/teacher.json",  "teacher entity"),
    ("entities/school.json",   "school entity"),
    ("entities/concept.json",  "concept entity"),
    ("entities/user.json",     "user entity"),
]

for relpath, label in files:
    fullpath = os.path.join("MOOCCube", relpath)
    print(f"\n=== {label} ({relpath}) ===")
    try:
        with open(fullpath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(repr(line.strip()[:300]))
    except Exception as e:
        print(f"Error: {e}")
