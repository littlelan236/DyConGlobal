"""Versioned static action classifier. Never executes trajectory commands."""

from __future__ import annotations

import ast
import json
import posixpath
import re
import warnings

import tree_sitter_bash
from tree_sitter import Language, Parser

VERSION = "observable-action-v1.1-draft"
ORDER = ("DELIVER", "EDIT", "ENV", "RUN", "CLEAN", "INSPECT")
LABELS = (*ORDER, "OTHER")
MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
READERS = {
    "cat",
    "grep",
    "rg",
    "egrep",
    "fgrep",
    "head",
    "tail",
    "ls",
    "find",
    "pwd",
    "wc",
    "diff",
    "awk",
    "file",
    "xxd",
    "od",
    "stat",
    "du",
    "df",
    "which",
    "whereis",
    "readlink",
    "realpath",
    "sort",
    "uniq",
    "cut",
    "tr",
    "strings",
    "less",
    "more",
}
FILTERS = {"head", "tail", "sort", "uniq", "cut", "tr", "wc", "tee"}
NEUTRAL = {"cd", "export", "unset", "true", "false", ":", "echo", "printf", "read"}
RUNNERS = {"pytest", "py.test", "tox", "nox", "pylint", "mypy", "flake8", "ruff", "sphinx-build"}
PACKAGES = {"pip", "pip3", "apt", "apt-get", "conda", "mamba", "uv"}


def shell_literal(value):
    """Decode a POSIX shell word without Python string-escape interpretation."""
    result, quote, i = [], None, 0
    while i < len(value):
        ch = value[i]
        if quote == "'":
            if ch == "'":
                quote = None
            else:
                result.append(ch)
        elif ch == "\\" and quote != "'":
            if i + 1 < len(value):
                following = value[i + 1]
                if quote is None or following in '$`"\\\n':
                    if following != "\n":
                        result.append(following)
                    i += 1
                else:
                    result.append(ch)
            else:
                result.append(ch)
        elif ch == quote:
            quote = None
        elif quote is None and ch in ("'", '"'):
            quote = ch
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _const(node, values):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id)
    return None


def python_behavior(source):
    """Recognize explicit file-writing and package-install syntax, not comments.

    This is deliberately not a Python effect system. Unknown calls can have
    arbitrary side effects, so the result also records that limitation.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {"OTHER"}, ["python_parse_error"], []
    values = {}
    changes = []
    labels = set()
    reasons = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = _const(node.value, values)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
        )
        if name == "open":
            mode = _const(node.args[1], values) if len(node.args) > 1 else "r"
            for kw in node.keywords:
                if kw.arg == "mode":
                    mode = _const(kw.value, values)
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                labels.add("EDIT")
                reasons.append("python_open_write_mode")
                if node.args:
                    path = _const(node.args[0], values)
                    if isinstance(path, str):
                        changes.append(path)
        if name in {"write_text", "write_bytes", "writelines"}:
            labels.add("EDIT")
            reasons.append("python_explicit_write")
        if name in {"exec", "eval", "system", "popen", "Popen", "check_call", "check_output", "run"}:
            # Do not guess the effect of nested dynamically constructed code.
            reasons.append("nested_dynamic_execution")
    if not labels:
        labels.add("RUN")
        reasons.append("python_execution")
    return labels, reasons, changes


class Classifier:
    """One classifier per rollout; remembers literal helper source from prior steps."""

    def __init__(self, cwd="/testbed"):
        self.parser = Parser(Language(tree_sitter_bash.language()))
        self.scripts = {}
        self.initial_cwd = cwd
        self.cwd = cwd

    def path(self, name):
        return posixpath.normpath(name if name.startswith("/") else posixpath.join(self.cwd, name))

    @staticmethod
    def text(node, data):
        return data[node.start_byte : node.end_byte].decode("utf-8")

    def classify(self, command):
        self.cwd = self.initial_cwd  # Each tool call uses a fresh shell.
        data = command.encode("utf-8")
        tree = self.parser.parse(data)
        if tree.root_node.has_error:
            return {
                "label": "OTHER",
                "components": [],
                "rules": ["shell_parse_error"],
                "uncertainties": ["shell_parse_error"],
                "compound": False,
            }
        labels, reasons, warnings = set(), [], []
        atoms = []
        new_scripts = {}

        def raw(n):
            return self.text(n, data)

        def descend(n):
            # Do not reinterpret quoted Python source/heredoc bodies as shell.
            if n.type == "command":
                atoms.append(n)
                return
            if n.type in {"string", "raw_string", "heredoc_body", "comment"}:
                return
            for k in n.named_children:
                descend(k)

        descend(tree.root_node)
        for n in atoms:
            name = n.child_by_field_name("name")
            if name is None:
                labels.add("OTHER")
                reasons.append("missing_command_name")
                continue
            name = shell_literal(raw(name))
            args = [
                shell_literal(raw(k))
                for k in n.named_children
                if k.type not in {"command_name", "variable_assignment"}
            ]
            redirects, bodies = [], []
            parent = n.parent
            while parent and parent.type not in {"program", "pipeline"}:
                if parent.type == "redirected_statement":
                    body = parent.child_by_field_name("body")
                    # Redirections on an and-list apply to its final command.
                    commands = [
                        a
                        for a in atoms
                        if body and a.start_byte >= body.start_byte and a.end_byte <= body.end_byte
                    ]
                    if commands and commands[-1].start_byte == n.start_byte:
                        for r in parent.named_children:
                            if r.type == "file_redirect":
                                dest = r.child_by_field_name("destination")
                                # 2>&1 is an FD dup, not a file creation.
                                op = raw(r)
                                if dest and ">" in op and ">&" not in op:
                                    target = shell_literal(raw(dest))
                                    if target not in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
                                        redirects.append(target)
                            if r.type == "heredoc_redirect":
                                bodies.extend(raw(k) for k in r.named_children if k.type == "heredoc_body")
                parent = parent.parent
            while posixpath.basename(name) in {"timeout", "env", "command", "nice"} and args:
                kind = posixpath.basename(name)
                if kind == "timeout":
                    while args and args[0].startswith("-"):
                        args.pop(0)
                    if args:
                        args.pop(0)
                elif kind == "env":
                    while args and ("=" in args[0] or args[0].startswith("-")):
                        args.pop(0)
                elif kind == "nice":
                    if args[:1] == ["-n"]:
                        args = args[2:]
                if not args:
                    break
                name, args = args[0], args[1:]
            head = posixpath.basename(name)
            atomic = set()
            rule = ""
            if head == "cd":
                if args and not re.search(r"[$`]", args[0]):
                    self.cwd = self.path(args[0])
            elif head in {"echo", "printf"} and MARKER in args and not redirects:
                atomic.add("DELIVER")
                rule = "protocol_marker_emission"
            elif head == "git":
                ga = list(args)
                while ga and ga[0].startswith("-"):
                    opt = ga.pop(0)
                    if opt in {"-C", "-c", "--git-dir", "--work-tree"} and ga:
                        ga.pop(0)
                op = ga[0] if ga else ""
                if op == "diff" and redirects:
                    atomic.add("DELIVER")
                    rule = "patch_export"
                elif op in {"add", "commit", "format-patch"}:
                    atomic.add("DELIVER")
                    rule = "git_package"
                elif op in {"checkout", "restore", "reset", "stash", "apply", "am", "revert"}:
                    atomic.add("EDIT")
                    rule = "git_worktree_change"
                    # Restores can invalidate previously remembered helper files.
                    self.scripts.clear()
                elif op == "clean":
                    atomic.add("CLEAN")
                    rule = "git_clean"
                elif op in {
                    "diff",
                    "status",
                    "show",
                    "log",
                    "blame",
                    "ls-files",
                    "rev-parse",
                    "tag",
                    "branch",
                }:
                    atomic.add("INSPECT")
                    rule = "git_inspection"
                else:
                    atomic.add("OTHER")
                    rule = "unsupported_git_operation"
            elif head in {"sed", "perl"} and any(a == "-i" or a.startswith("-i") for a in args):
                atomic.add("EDIT")
                rule = "in_place_edit"
            elif head == "patch":
                atomic.add("EDIT")
                rule = "patch_apply"
            elif head in {"cp", "mv", "touch", "mkdir", "install"}:
                atomic.add("EDIT")
                rule = "filesystem_construct"
                for a in args:
                    if not a.startswith("-"):
                        self.scripts.pop(self.path(a), None)
            elif head in {"rm", "rmdir", "unlink"}:
                atomic.add("CLEAN")
                rule = "filesystem_remove"
                for a in args:
                    if not a.startswith("-"):
                        self.scripts.pop(self.path(a), None)
            elif head in PACKAGES:
                if any(
                    a in {"install", "uninstall", "remove", "update", "upgrade", "create", "sync"}
                    for a in args
                ):
                    atomic.add("ENV")
                    rule = "dependency_management"
                else:
                    atomic.add("INSPECT")
                    rule = "dependency_inspection"
            elif head in {"chmod", "chown", "source"}:
                atomic.add("ENV")
                rule = "runtime_environment"
            elif re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", head):
                pa = list(args)
                while pa and pa[0].startswith("-") and pa[0] not in {"-c", "-m", "-"}:
                    opt = pa.pop(0)
                    if opt in {"-X", "-W"} and pa:
                        pa.pop(0)
                src = None
                if pa[:1] == ["-c"] and len(pa) > 1:
                    src = pa[1]
                elif bodies and (not pa or pa[0] == "-"):
                    src = "\n".join(bodies)
                elif pa[:1] == ["-m"]:
                    module = pa[1] if len(pa) > 1 else ""
                    if module in {"pip", "pip3", "venv", "ensurepip", "build"}:
                        atomic.add(
                            "ENV" if module != "pip" or "install" in pa or "uninstall" in pa else "INSPECT"
                        )
                        rule = "python_environment_module"
                    else:
                        atomic.add("RUN")
                        rule = "python_module_execution"
                elif pa and not pa[0].startswith("-"):
                    target = self.path(pa[0])
                    src = new_scripts.get(target, self.scripts.get(target))
                    if src is None:
                        atomic.add("RUN")
                        rule = "script_invocation_source_unavailable"
                        warnings.append("script_source_unavailable")
                elif not bodies:
                    atomic.add("OTHER")
                    rule = "python_input_unavailable"
                if src is not None:
                    result, why, _ = python_behavior(src)
                    atomic.update(result)
                    reasons.extend(why)
                    if "nested_dynamic_execution" in why:
                        warnings.append("nested_dynamic_execution")
                    if "python_parse_error" in why:
                        warnings.append("python_parse_error")
                    rule = "python_ast"
            elif head in RUNNERS:
                atomic.add("EDIT" if head == "ruff" and "--fix" in args else "RUN")
                rule = "test_or_check_runner"
            elif head in {"make", "cmake", "ninja", "gcc", "g++", "npm"}:
                atomic.add("ENV")
                rule = "build_environment"
            elif head == "xargs":
                if any(a in {"grep", "rg", "cat", "head", "sed"} for a in args):
                    atomic.add("INSPECT")
                    rule = "xargs_reader"
                else:
                    atomic.add("OTHER")
                    rule = "unsupported_xargs"
            elif head in {"curl", "wget"}:
                atomic.add("OTHER")
                rule = "network_effects_not_proven_readonly"
            elif head == "find" and any(a in {"-delete", "-exec", "-execdir", "-ok", "-okdir"} for a in args):
                atomic.add("OTHER")
                rule = "find_effects_not_supported"
            elif head in READERS or head == "sed":
                atomic.add("INSPECT")
                rule = "text_or_path_inspection"
            elif head not in NEUTRAL:
                atomic.add("OTHER")
                rule = "unsupported_command"
            if redirects and head in READERS | {"echo", "printf", "sed", "perl"}:
                atomic.discard("INSPECT")
                atomic.add("EDIT")
                rule = "explicit_file_construction"
                for target in redirects:
                    key = self.path(target)
                    self.scripts.pop(key, None)
                    if head == "cat" and bodies:
                        new_scripts[key] = "\n".join(bodies)
            # Filters are output shaping rather than a second purpose.
            if head in FILTERS and n.parent and n.parent.type == "pipeline":
                earlier = [
                    a for a in atoms if a.start_byte < n.start_byte and a.start_byte >= n.parent.start_byte
                ]
                if earlier and atomic == {"INSPECT"}:
                    atomic.clear()
            labels.update(atomic)
            if rule:
                reasons.append(rule)
        self.scripts.update(new_scripts)
        if "OTHER" in labels or not labels:
            label = "OTHER"
        else:
            label = next(k for k in ORDER if k in labels)
        if label == "OTHER":
            warnings.append("unclassified_or_unsupported")
        return {
            "label": label,
            "components": [k for k in LABELS if k in labels],
            "rules": sorted(set(reasons)),
            "uncertainties": sorted(set(warnings)),
            "compound": len(labels - {"OTHER"}) > 1,
        }


def extract_command(step):
    calls = step.get("tool_calls") or []
    if len(calls) != 1:
        return None
    function = calls[0].get("function") or {}
    if function.get("name") != "bash":
        return None
    args = function.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return None
    return args.get("command") if isinstance(args, dict) else None


def classify_action_evidence(evidence, classifier):
    """Classify tool-call evidence from one assistant message without executing it."""
    step = {
        "tool_calls": (evidence or {}).get("tool_calls") or [],
    }
    command = extract_command(step)
    if not isinstance(command, str):
        return {
            "label": "OTHER",
            "components": [],
            "rules": ["unsupported_tool_shape"],
            "uncertainties": ["unsupported_tool_shape"],
            "compound": False,
            "command": command,
        }
    return {"command": command, **classifier.classify(command)}
