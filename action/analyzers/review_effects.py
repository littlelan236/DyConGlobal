"""Static shell/Python write-intent features; never execute trajectory code.

Reuses the action classifier's shell parser. These are bounded static features,
not a proof of all effects of arbitrary Python or shell programs.
"""
from __future__ import annotations
import ast
import posixpath
import re
import warnings as python_warnings
from action.analyzers.taxonomy import Classifier, shell_literal

def literal(node, values):
    if isinstance(node,ast.Constant):return node.value
    if isinstance(node,ast.Name):return values.get(node.id)
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):
        left,right=literal(node.left,values),literal(node.right,values)
        if isinstance(left,str) and isinstance(right,str):return left+right
    if isinstance(node,ast.Call):
        name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
        args=[literal(x,values) for x in node.args]
        if name in ('Path','str') and args:return args[0]
        if name=='join' and all(isinstance(x,str) for x in args):return posixpath.join(*args)
    return None


def python_writes(source):
    try:
        with python_warnings.catch_warnings():
            python_warnings.simplefilter('ignore',SyntaxWarning)
            tree=ast.parse(source)
    except (SyntaxError,ValueError):return [],['python_parse_error']
    values={};writes=[];warnings=[]
    for _ in range(2):
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name):values[target.id]=literal(node.value,values)
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
        if name=='open':
            mode=literal(node.args[1],values) if len(node.args)>1 else 'r'
            for kw in node.keywords:
                if kw.arg=='mode':mode=literal(kw.value,values)
            if isinstance(mode,str) and any(x in mode for x in 'wax+'):
                path=literal(node.args[0],values) if node.args else None
                if isinstance(path,str):writes.append(path)
                else:warnings.append('python_dynamic_write_path')
        if name in ('write_text','write_bytes') and isinstance(node.func,ast.Attribute):
            path=literal(node.func.value,values)
            if isinstance(path,str):writes.append(path)
            else:warnings.append('python_dynamic_write_path')
        if name in ('exec','eval','system','popen','Popen','run','check_call','check_output'):
            warnings.append('nested_execution_not_resolved')
    return sorted(set(writes)),sorted(set(warnings))


class Effects:
    def __init__(self,cwd='/testbed'):
        self.initial=cwd;self.scripts={};self.parser=Classifier(cwd=cwd).parser

    def parse(self,command):
        if not isinstance(command,str):return {'writes':[],'deletes':[],'state_changes':[],'warnings':['no_supported_command']}
        raw=command.encode();tree=self.parser.parse(raw)
        if tree.root_node.has_error:return {'writes':[],'deletes':[],'state_changes':[],'warnings':['shell_parse_error']}
        text=lambda n:raw[n.start_byte:n.end_byte].decode()
        atoms=[]
        def walk(n):
            if n.type=='command':atoms.append(n);return
            if n.type in ('string','raw_string','heredoc_body','comment'):return
            for k in n.named_children:walk(k)
        walk(tree.root_node)
        cwd=self.initial;writes=[];deletes=[];states=[];warnings=[]
        def path(value):return posixpath.normpath(value if value.startswith('/') else posixpath.join(cwd,value))
        for node in atoms:
            name=node.child_by_field_name('name')
            if name is None:continue
            name=posixpath.basename(shell_literal(text(name)))
            args=[shell_literal(text(k)) for k in node.named_children if k.type not in ('command_name','variable_assignment')]
            redirects=[];bodies=[];parent=node.parent
            while parent and parent.type not in ('program','pipeline'):
                if parent.type=='redirected_statement':
                    body=parent.child_by_field_name('body')
                    related=[n for n in atoms if body and n.start_byte>=body.start_byte and n.end_byte<=body.end_byte]
                    if related and related[-1].start_byte==node.start_byte:
                        for r in parent.named_children:
                            if r.type=='file_redirect':
                                dest=r.child_by_field_name('destination');op=text(r)
                                if dest and '>' in op and '>&' not in op:
                                    target=shell_literal(text(dest))
                                    if target not in ('/dev/null','/dev/stdout','/dev/stderr'):redirects.append(path(target))
                            if r.type=='heredoc_redirect':bodies += [text(k) for k in r.named_children if k.type=='heredoc_body']
                parent=parent.parent
            if name=='cd' and args:cwd=path(args[0]);continue
            writes+=redirects
            for target in redirects:
                if name=='cat' and bodies:self.scripts[target]='\n'.join(bodies)
                else:self.scripts.pop(target,None)
            if name in ('sed','perl') and any(x.startswith('-i') for x in args):
                if args and not args[-1].startswith('-'):writes.append(path(args[-1]))
                else:warnings.append('inplace_target_unknown')
            elif name in ('cp','mv','touch','install'):
                targets=[a for a in args if not a.startswith('-')]
                if targets:
                    destination = path(targets[-1])
                    writes.append(destination)
                    if name in ('cp', 'mv') and len(targets) > 1:
                        source = path(targets[-2])
                        if source in self.scripts:
                            self.scripts[destination] = self.scripts[source]
                        if name == 'mv':
                            deletes.append(source)
                            self.scripts.pop(source, None)
            elif name in ('rm','rmdir','unlink'):
                removed = [path(a) for a in args if not a.startswith('-')]
                deletes += removed
                for target in list(self.scripts):
                    if any(target == p or target.startswith(p.rstrip('/') + '/') for p in removed):
                        self.scripts.pop(target, None)
            elif name=='git':
                ga=args[:]
                while ga and ga[0].startswith('-'):
                    opt=ga.pop(0)
                    if opt in ('-C','-c','--git-dir','--work-tree') and ga:ga.pop(0)
                if ga and ga[0] in ('checkout','restore','reset','stash','apply','am','revert') and not (ga[0] == 'stash' and len(ga) > 1 and ga[1] in ('list','show')):
                    states.append('git_'+ga[0]);self.scripts.clear()
            elif name=='patch':states.append('patch_apply')
            elif re.fullmatch(r'python(?:\d+(?:\.\d+)*)?',name):
                src=None
                args = list(args)
                while args and args[0] not in ('-c', '-m', '-') and args[0].startswith('-'):
                    option = args.pop(0)
                    if option in ('-W', '-X') and args:
                        args.pop(0)
                if '-c' in args:
                    j=args.index('-c');src=args[j+1] if len(args)>j+1 else None
                elif bodies:src='\n'.join(bodies)
                elif args and '-m' not in args:
                    target=path(args[0]);src=self.scripts.get(target)
                    if src is None:warnings.append('script_effects_unavailable')
                if src is not None:
                    paths,why=python_writes(src);writes += [path(p) for p in paths];warnings+=why
            elif name in ('bash','sh'):
                warnings.append('nested_shell_effects_not_resolved')
            elif name == 'xargs' and not any(a in ('cat','grep','rg','head','tail','sed') for a in args):
                warnings.append('nested_shell_effects_not_resolved')
        if any(re.search(r'[$`]', p) for p in writes + deletes):
            warnings.append('dynamic_shell_path')
        return {'writes':sorted(set(writes)),'deletes':sorted(set(deletes)),
                'state_changes':sorted(set(states)),'warnings':sorted(set(warnings))}

