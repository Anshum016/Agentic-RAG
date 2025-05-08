HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

import requests
import base64

def get_all_files(owner, repo, path=""):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url, headers=HEADERS)
    
    all_contents = {}

    if response.status_code == 200:
        items = response.json()
        for item in items:
            if item['type'] == 'file':
                file_content = get_file_content(owner, repo, item['path'])
                all_contents[item['path']] = file_content
            elif item['type'] == 'dir':
                sub_contents = get_all_files(owner, repo, item['path'])
                all_contents.update(sub_contents)
    return all_contents

def get_file_content(owner, repo, file_path):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        content_base64 = data['content']
        encoding = data.get('encoding', '')

        if encoding == 'base64':
            try:
                decoded_content = base64.b64decode(content_base64)
                try:
                    return decoded_content.decode('utf-8')  # text file
                except UnicodeDecodeError:
                    return decoded_content  # binary file
            except Exception as e:
                print(f" Error decoding {file_path}: {e}")
                return None
        else:
            print(f"Unsupported encoding for {file_path}")
            return None
    else:
        print(f" Failed to fetch {file_path}: {response.status_code}")
        return None

repo_owner = "feder-cr"
repo_name = "Jobs_Applier_AI_Agent_AIHawk"

all_files = get_all_files(repo_owner, repo_name)

for path, content in all_files.items():
    print(f"\n--- {path} ---\n{content}")

