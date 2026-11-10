from core.io import hash_json
SCHEMA_VERSION=1
MASK_POLICY="native_completion_reasoning_action_v1"
def tokens(value,name):
    if not isinstance(value,list) or not value or any(type(x) is not int or x<0 or x>=2**32 for x in value): raise ValueError(f"Invalid native {name} tokens")
    return value
def validate(prompt,completion,prompt_sha,output_sha,*,verify_hashes=True):
    tokens(prompt,"prompt"); tokens(completion,"completion")
    if verify_hashes and (hash_json(prompt)!=prompt_sha or hash_json(completion)!=output_sha): raise ValueError("Native token checksum mismatch")
def group_id(row): return "g-"+hash_json({k:row[k] for k in ("source_run_id","rollout_id","session_id","decision_id","prompt_sha256")})[:40]
def branch_id(gid,bid,request): return "b-"+hash_json([gid,bid,request])[:40]
def spans(prompt_len,completion_len):
    if prompt_len<1 or completion_len<1: raise ValueError("Empty prompt/completion")
    return [[prompt_len,prompt_len+completion_len]]
