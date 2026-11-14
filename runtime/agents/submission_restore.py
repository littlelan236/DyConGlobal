"""Bounded, authenticated restoration of declared workspace outputs into a sandbox."""
from __future__ import annotations

import hashlib
import posixpath
import tarfile
from pathlib import Path, PurePosixPath

from application.execution.subject import safe_output_path
from core.io import hash_json, sha256

MAX_EXPANDED = 8 * 1024**3
MAX_MEMBERS = 250000


def archive_inventory(path, expected, source):
    """Authenticate every regular member and reject extraction through links."""
    source = safe_output_path(source)
    prefix = PurePosixPath(source).name
    if sha256(path) != expected:
        raise ValueError('Changed subject archive')
    entries, seen, links, expanded = [], set(), set(), 0
    with tarfile.open(path, 'r|*') as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if (len(entries) >= MAX_MEMBERS or name.is_absolute() or '..' in name.parts
                    or not name.parts or name.parts[0] != prefix or str(name) in seen):
                raise ValueError('Unsafe or duplicate archive member: ' + member.name)
            seen.add(str(name))
            row = {'name': str(name), 'mode': member.mode & 0o777}
            if member.isdir():
                row['kind'] = 'directory'
            elif member.isfile():
                expanded += member.size
                if expanded > MAX_EXPANDED:
                    raise ValueError('Expanded subject archive budget exceeded')
                h = hashlib.sha256()
                file = stream.extractfile(member)
                size = 0
                for block in iter(lambda: file.read(1024**2), b''):
                    size += len(block)
                    h.update(block)
                if size != member.size:
                    raise ValueError('Incomplete archive member')
                row.update(kind='file', size=size, sha256=h.hexdigest())
            elif member.issym() or member.islnk():
                target = (posixpath.normpath(posixpath.join(posixpath.dirname(str(name)), member.linkname))
                          if member.issym() else posixpath.normpath(member.linkname))
                # /skills is the existing read-only task skill mount, never extracted into.
                skill_link = str(name) == 'workspace/skills' and member.linkname == '/skills'
                if not skill_link and (not PurePosixPath(target).parts or PurePosixPath(target).is_absolute()
                                       or PurePosixPath(target).parts[0] != prefix
                                       or '..' in PurePosixPath(target).parts):
                    raise ValueError('Archive link escapes declared output: ' + str(name))
                row.update(kind='symlink' if member.issym() else 'hardlink', target=member.linkname)
                links.add(str(name))
            else:
                raise ValueError('Special archive member: ' + str(name))
            entries.append(row)
    for row in entries:
        if any(str(parent) in links for parent in PurePosixPath(row['name']).parents):
            raise ValueError('Archive member has a link parent: ' + row['name'])
    if sha256(path) != expected:
        raise ValueError('Subject archive changed while reading')
    return {'source_path': source, 'sha256': expected, 'entries': entries,
            'expanded_bytes': expanded, 'inventory_sha256': hash_json(entries)}


# Runs only inside the fresh owned task container, using the published CPython.
RESTORE_CODE = r'''
import hashlib,json,os,pathlib,shutil,sys,tarfile,tempfile
request=json.loads(pathlib.Path(sys.argv[1]).read_text())
source=pathlib.Path(request['source_path'])
assert str(source) in request['allowed_paths'] and source.is_absolute() and len(source.parts)>1
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
if request['kind']=='missing':
 if source.is_symlink() or source.is_file():source.unlink()
 elif source.is_dir():shutil.rmtree(source)
 print(json.dumps({'verified':not source.exists(),'missing':str(source)}))
 sys.exit(0)
archive=pathlib.Path(request['uploaded_path'])
assert digest(archive)==request['sha256']
if request['kind']=='file':
 source.parent.mkdir(parents=True,exist_ok=True)
 if source.is_symlink():source.unlink()
 shutil.copyfile(archive,source);os.chmod(source,int(request['mode'],8)&0o777)
 assert digest(source)==request['sha256']
 print(json.dumps({'verified':True,'files':1,'source_path':str(source)}))
 sys.exit(0)
entries={r['name']:r for r in request['entries']}
with tempfile.TemporaryDirectory(prefix='dycon-regrade-') as stage:
 stage=pathlib.Path(stage)
 with tarfile.open(archive,'r|*') as stream:
  for member in stream:
   name=str(pathlib.PurePosixPath(member.name));row=entries[name];target=stage/name
   assert target.resolve().is_relative_to(stage)
   if row['kind']=='directory':target.mkdir(parents=True,exist_ok=True)
   elif row['kind']=='file':
    target.parent.mkdir(parents=True,exist_ok=True)
    with stream.extractfile(member) as src,open(target,'wb') as dst:shutil.copyfileobj(src,dst)
    assert target.stat().st_size==row['size'] and digest(target)==row['sha256']
    os.chmod(target,row['mode'])
 for row in request['entries']:
  target=stage/row['name']
  if row['kind']=='symlink':
   target.parent.mkdir(parents=True,exist_ok=True);os.symlink(row['target'],target)
  elif row['kind']=='hardlink':
   link=stage/row['target'];assert link.is_file() and not link.is_symlink()
   target.parent.mkdir(parents=True,exist_ok=True);os.link(link,target)
 for row in reversed(request['entries']):
  if row['kind']=='directory':os.chmod(stage/row['name'],row['mode'])
 if source.is_symlink() or source.is_file():source.unlink()
 elif source.is_dir():shutil.rmtree(source)
 source.parent.mkdir(parents=True,exist_ok=True)
 shutil.move(str(stage/source.name),str(source))
for row in request['entries']:
 path=source.parent/row['name']
 if row['kind']=='file':assert digest(path)==row['sha256']
 if row['kind']=='symlink':assert os.readlink(path)==row['target']
print(json.dumps({'verified':True,'source_path':str(source),'entries':len(entries),
 'files':sum(r['kind']=='file' for r in entries.values())}))
'''
