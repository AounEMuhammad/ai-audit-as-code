import json, os
def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w') as f: f.write(json.dumps(data, indent=2))
