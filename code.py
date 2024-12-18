python -c "
import json
old = json.load(open('$combined_json'))
for line in '''$result'''.split('\n'):
    line = line.strip()
    if line.startswith('{') and line.endswith('}'):
        old.append(json.loads(line))
json.dump(old, open('$combined_json','w'), indent=2)
"

python -c "import json; old=json.load(open('$combined_json')); lines='''$result'''.split('\n'); 
filtered=[json.loads(l.strip()) for l in lines if l.strip().startswith('{') and l.strip().endswith('}')]; 
old.extend(filtered); json.dump(old, open('$combined_json','w'), indent=2)"
