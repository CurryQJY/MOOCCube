import json

def peek(filepath):
    print(f"\n--- {filepath} ---")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            if line.startswith('{'):
                data = json.loads(line)
                print(json.dumps({k: data[k] for k in list(data.keys())[:3]}, ensure_ascii=False, indent=2))
            else:
                print(line[:200])
    except Exception as e:
        print(f"Error: {e}")

peek('MOOCCube/entities/course.json')
peek('MOOCCube/relations/course-concept.json')
peek('MOOCCube/relations/teacher-course.json')
peek('MOOCCube/relations/school-course.json')
peek('MOOCCube/relations/prerequisite-dependency.json')
peek('MOOCCube/entities/user.json')
