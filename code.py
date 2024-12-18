python -c "
import json
old = json.load(open('$combined_json'))
for line in '''$result'''.split('\n'):
    line = line.strip()
    if line.startswith('{') and line.endswith('}'):
        old.append(json.loads(line))
json.dump(old, open('$combined_json','w'), indent=2)
"

