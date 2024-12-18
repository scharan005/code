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

Traceback (most recent call last):

 File "/proj/zendnn/chponnad/new_dl/dlrm/dlrm/dlrm_s_pytorch_export.py", line 876, in <module>

  run()

 File "/proj/zendnn/chponnad/new_dl/dlrm/dlrm/dlrm_s_pytorch_export.py", line 864, in run

  inference(

 File "/proj/zendnn/chponnad/new_dl/dlrm/dlrm/dlrm_s_pytorch_export.py", line 509, in inference

  export_dlrm_mod = dlrm_export_model(

 File "/proj/zendnn/chponnad/new_dl/dlrm/dlrm/dlrm_s_pytorch_export.py", line 155, in dlrm_export_model

  so_path = torch._export.aot_compile(

 File "/proj/zendnn/chponnad/anaconda3/envs/torch_export_620/lib/python3.9/site-packages/torch/_export/__init__.py", line 274, in aot_compile

  gm = _export_to_torch_ir(

 File "/proj/zendnn/chponnad/anaconda3/envs/torch_export_620/lib/python3.9/site-packages/torch/export/_trace.py", line 576, in _export_to_torch_ir

  raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e)) # noqa: B904

torch._dynamo.exc.UserError: L['input'][9].size()[0] = 26 is not equal to L['input'][0].size()[0] = 1
