import sys,json,datetime
from pathlib import Path
sys.path.insert(0,'/home/wangtianyi/DyConGlobal')
from core.model_identity import register
from baseline.catalog import Catalog
from core.io import write_json
root=Path('/home/wangtianyi/DyConGlobal')
models={}
for name in ['Qwen3.5-9B','Qwen3.5-27B','Qwen3.6-27B','Qwen3.6-35B-A3B']:
 path=Path('/home/wangtianyi/tactRefractor/assets/models')/name
 fp=register(root,path)
 models[name]={'path':str(path),'fingerprint':fp}
 print(name,fp,flush=True)
write_json(root/'state/extraction-models.json',models)
with Catalog(root).transaction() as m:
 for row in m['rollouts'].values():
  name=row['provenance']['legacy_model_identity']['model_name']
  row['provenance'].setdefault('import_model_fingerprint',row['model_fingerprint'])
  row['model_fingerprint']=models[name]['fingerprint']
  row['provenance']['replay_checkpoint']={**models[name],'binding_basis':'legacy recorded model name mapped to available checkpoint; checkpoint bytes hashed, historical serving bytes not independently attested'}
print('BOUND',flush=True)
