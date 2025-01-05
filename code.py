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

if hasattr(m, "module"):  
        print("Applying zentorch optimization to `m.module`")
        m.module = torch.compile(m.module, backend="zentorch")  # Apply IPEX optimization
    elif hasattr(m, "model"):  
        print("Applying zentorch optimization to `m.model`")
        m.model = torch.compile(m.model, backend="zentorch")  # Apply IPEX optimization
    else:
        print("Warning: Model object does not have 'module' or 'model' attribute. Applying to full object.")
        m = torch.compile(m, backend="zentorch")
