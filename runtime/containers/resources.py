"""Immutable supplemental test assets at the shared container boundary."""
from pathlib import Path
import json, shlex
from core.io import sha256


def assets(cfg):
    rows = json.loads(cfg["environment"].get("variables", {}).get("DYCON_SWEBENCH_RESOURCES", "[]"))
    root = (Path(cfg["project_root"]) / "runtime/resources").resolve()
    targets = set()
    for row in rows:
        source = Path(row["source"]).resolve()
        target = Path(row["destination"])
        if (not source.is_relative_to(root) or not source.is_file()
                or sha256(source) != row["sha256"] or target in targets
                or not target.is_relative_to("/testbed") or ".." in target.parts
                or target.suffix not in {".txt", ".csv", ".dat", ".dcm", ".gz", ".zip", ".json", ".epw", ".h5", ".bz2", ".pkz"}):
            raise ValueError("Invalid immutable test resource")
        if row.get('transform') not in (None, 'sklearn-california-v1'):
            raise ValueError('Unsupported resource transformation')
        if row.get('transform') and (target.name != 'cal_housing_py3.pkz'
                or row.get('sha256') != 'aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681'
                or not row.get('loader_sha256')):
            raise ValueError('Invalid California housing source contract')
        targets.add(target)
    return rows


def mounts(cfg):
    return {row["source"]: {"bind": "/opt/dycon-resources/" + row["sha256"], "mode": "ro"}
            for row in assets(cfg)}


def preparation(cfg):
    rows = assets(cfg)
    if not rows:
        return ""
    code = """import hashlib,json,os
rows=json.loads(ASSETS)
for row in rows:
    source=os.path.join('/opt/dycon-resources',row['sha256'])
    with open(source,'rb') as stream:data=stream.read()
    if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Mounted resource digest mismatch')
    target=row['destination']
    if os.path.islink(target) or not os.path.realpath(target).startswith('/testbed/'):raise ValueError('Resource target escapes testbed')
    if row.get('transform')=='sklearn-california-v1':
        import tarfile,numpy as np
        from sklearn.datasets import california_housing as loader
        from sklearn.utils import _joblib
        with open(loader.__file__,'rb') as stream:loader_bytes=stream.read()
        if hashlib.sha256(loader_bytes).hexdigest()!=row['loader_sha256']:raise ValueError('California housing loader changed')
        if loader.ARCHIVE.checksum!=row['sha256']:raise ValueError('California housing official checksum changed')
        parent=os.path.dirname(target)
        if loader._pkl_filepath(parent,'cal_housing.pkz')!=target:raise ValueError('Unexpected official cache path')
        with tarfile.open(source,'r:gz') as archive:
            member=archive.getmember('CaliforniaHousing/cal_housing.data')
            if not member.isfile() or member.size>20000000:raise ValueError('Invalid California housing member')
            matrix=np.loadtxt(archive.extractfile(member),delimiter=',')
        if matrix.shape!=(20640,9) or not np.isfinite(matrix).all():raise ValueError('Invalid California housing matrix')
        matrix=matrix[:,[8,7,2,3,4,5,6,1,0]]
        if os.path.exists(target):
            if not np.array_equal(_joblib.load(target),matrix):raise ValueError('Conflicting original resource cache')
        else:
            if not os.path.isdir(parent):os.makedirs(parent)
            _joblib.dump(matrix,target,compress=6)
        print('DYCON_RESOURCE_CACHE_READY',target,row['sha256'],row['loader_sha256'])
        continue
    if os.path.exists(target):
        with open(target,'rb') as stream:original=stream.read()
        if original!=data:raise ValueError('Conflicting original resource')
    parent=os.path.dirname(target)
    if not os.path.isdir(parent):os.makedirs(parent)
    with open(target,'wb') as stream:stream.write(data)
    print('DYCON_RESOURCE_READY',row['destination'],row['sha256'])
"""
    code = "ASSETS=" + repr(json.dumps(rows)) + "\n" + code
    return "python -c " + shlex.quote(code)
