"""Frozen task-container package sources shared by rollout and verifier."""
import re
import shlex
from pathlib import Path

from core.io import sha256


def task_dependency_overlay(dycon, network_mode):
    """Return read-only task wheels and package-index variables for one task."""
    mounts = []
    environment = {}
    preparation = dycon.get("task_preparation") or {}
    if preparation.get("python_constraints"):
        environment["PIP_CONSTRAINT"] = "/opt/dycon-task-constraints.txt"
        environment["UV_CONSTRAINT"] = "/opt/dycon-task-constraints.txt"
    wheelhouse = dycon.get("task_wheelhouse")
    if wheelhouse:
        source = Path(wheelhouse).resolve()
        manifest = source / "manifest.json"
        expected = dycon.get("task_wheelhouse_manifest_sha256")
        if not source.is_dir() or not manifest.is_file() or sha256(manifest) != expected:
            raise ValueError("Harbor task wheelhouse identity changed after run creation")
        mounts.append({"type": "bind", "source": str(source),
                       "target": "/opt/dycon-task-wheels", "read_only": True})
        environment.update(PIP_FIND_LINKS="/opt/dycon-task-wheels",
                           UV_FIND_LINKS="/opt/dycon-task-wheels")
    mode = getattr(network_mode, "value", network_mode)
    index = dycon.get("task_python_index")
    if index and mode == "public":
        environment.update(PIP_INDEX_URL=index, UV_INDEX_URL=index, UV_DEFAULT_INDEX=index)
    cache = dycon.get("task_resource_cache")
    if cache and mode == "public":
        bundle = Path(cache["ca_bundle"])
        if sha256(bundle) != cache["ca_sha256"]:
            raise ValueError("Resource cache trust bundle changed")
        mounts.append({"type": "bind", "source": str(bundle),
                       "target": "/opt/dycon-resource-cache-ca.pem", "read_only": True})
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            environment[key] = cache["endpoint"]
        environment.update(NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1", no_proxy="localhost,127.0.0.1,0.0.0.0,::1",
                           SSL_CERT_FILE="/opt/dycon-resource-cache-ca.pem",
                           SSL_CERT_DIR="/etc/ssl/certs", CURL_CA_BUNDLE="/opt/dycon-resource-cache-ca.pem",
                           REQUESTS_CA_BUNDLE="/opt/dycon-resource-cache-ca.pem",
                           GIT_SSL_CAINFO="/opt/dycon-resource-cache-ca.pem",
                           NODE_EXTRA_CA_CERTS="/opt/dycon-resource-cache-ca.pem",
                           UV_NATIVE_TLS="true", UV_HTTP_TIMEOUT="600", UV_HTTP_RETRIES="5",
                           PIP_DEFAULT_TIMEOUT="600", PIP_RETRIES="5",
                           HF_HUB_ETAG_TIMEOUT="120", HF_HUB_DOWNLOAD_TIMEOUT="600",
                           HF_HUB_DISABLE_XET="1", GIT_CONFIG_COUNT="2",
                           GIT_CONFIG_KEY_0="http.lowSpeedLimit", GIT_CONFIG_VALUE_0="1",
                           GIT_CONFIG_KEY_1="http.lowSpeedTime", GIT_CONFIG_VALUE_1="600")
    return mounts, environment


def apt_mirror_warmup_required(task):
    """Refresh Ubuntu indexes before setup only when solve.sh installs first."""
    script = Path(task["instance"]["task_path"]) / "solution/solve.sh"
    if not script.is_file():
        return False
    lines = []
    for line in script.read_text(errors="replace").splitlines():
        lines.append(line.split("#", 1)[0])
    text = "\n".join(lines)
    install = re.search(r"\b(?:apt-get|apt)\s+(?:-y\s+)?install\b", text)
    update = re.search(r"\b(?:apt-get|apt)\s+update\b", text)
    return bool(install and (update is None or install.start() < update.start()))


def apt_mirror_setup_command(mirror, *, warmup=False):
    """Rewrite only Ubuntu archive URIs and preserve signed apt metadata policy."""
    if not re.fullmatch(r"https://[A-Za-z0-9.-]+(?::[0-9]{1,5})?/[A-Za-z0-9._~/-]+", mirror):
        raise ValueError("APT mirror URL contains unsafe characters")
    mirror = mirror.rstrip("/") + "/"
    expression = (r"s#https?://([[:alnum:]-]+\.)?(archive|security)\.ubuntu\.com/ubuntu/?#"
                  + mirror + "#g")
    command = [
        "set -eu",
        'if [ ! -r /etc/os-release ]; then printf "skipped:no-os-release\\n"; exit 0; fi',
        ". /etc/os-release",
        'if [ "${ID:-}" != ubuntu ]; then printf "skipped:unsupported-distro:%s\\n" "${ID:-unknown}"; exit 0; fi',
        'found=0',
        'for source in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do',
        '  [ -f "$source" ] || continue',
        f'  sed -i -E {shlex.quote(expression)} "$source"',
        f'  if grep -Fq {shlex.quote(mirror)} "$source"; then found=1; fi',
        "done",
        'if [ "$found" -ne 1 ]; then printf "error:ubuntu-apt-uri-not-found\\n"; exit 74; fi',
    ]
    if warmup:
        command.append("apt-get update -o Acquire::Retries=2 -o Acquire::http::Timeout=30 "
                       "-o Acquire::https::Timeout=30")
    command.append(f'printf "applied:{mirror} warmup={int(warmup)}\\n"')
    return "\n".join(command)


def resource_cache_setup_command(cache):
    """Configure system TLS/APT before package installation."""
    proxy = cache["endpoint"]
    if any(c in proxy for c in '\n\r"\\'):
        raise ValueError("Invalid resource proxy URL")
    apt = ('Acquire::http::Proxy "' + proxy + '";\n' +
           'Acquire::https::Proxy "' + proxy + '";\n' +
           'Acquire::Retries "5";\n' +
           'Acquire::http::Timeout "120";\n' +
           'Acquire::https::Timeout "120";\n' +
           'Acquire::https::CaInfo "/opt/dycon-resource-cache-ca.pem";\n')
    return "\n".join([
        "set -eu", "mkdir -p /etc/ssl/certs",
        "cat /opt/dycon-resource-cache-ca.pem > /etc/ssl/certs/ca-certificates.crt",
        "if [ -d /etc/apt/apt.conf.d ]; then",
        "printf %s " + shlex.quote(apt) + " > /etc/apt/apt.conf.d/99dycon-resource-cache",
        "fi",
    ])
