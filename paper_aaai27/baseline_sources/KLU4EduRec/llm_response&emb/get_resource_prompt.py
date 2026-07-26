import os, sys
import argparse
import json

cur_path = os.path.abspath(os.path.dirname(__file__))
project_path = cur_path[:cur_path.find('LLM4ERec')] + 'LLM4ERec'
sys.path.append(project_path)
os.chdir(project_path)

def get_resource_prompt(dataset_name):
    summary_list = []

    resource_file = f'data/{dataset_name}/resource_info.json'
    with open(resource_file, 'r', encoding='utf-8') as f:
        for line in f:
            resource_info = json.loads(line.strip())
            summary_prompt = "Assume you are an educational content analysis expert. Below is the basic information of a specific resource.\n"+\
                            f"Resource Type: {resource_info['resource_type']}\n"+\
                            f"Resource Title: {resource_info['resource_title']}\n"+\
                            f"Resource Content: {resource_info['resource_content']}\n"+\
                            f"Belong Course: {resource_info.get('related_course', None)}\n"+\
                            f"Related Concepts: {resource_info.get('related_concepts', None)}\n"+\
                            "Please provide a concise summary that describe the core knowledge of the above resource, the summary should be no more than 80 words.\n"
            summary_list.append({resource_info['resource_id']: summary_prompt})

    summary_file = f'data/{dataset_name}/resource_summary_prompt.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        for item in summary_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Resource prompts for dataset '{dataset_name}' have been generated.")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='mooccubex')
    args = parser.parse_args()
    get_resource_prompt(args.dataset)