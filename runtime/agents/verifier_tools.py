"""Shared sandbox-local tools for official online and offline verifiers."""


def verifier_tool_command():
    # BenchFlow deliberately excludes answer workspaces from its trusted PATH.
    # Keep its invariants: install only the published curl shim in the existing
    # trusted /usr/local/bin, and let official scripts source the local uv env.
    return r'''set -eu
test ! -L /root/.local
test ! -L /root/.local/bin
mkdir -p /root/.local/bin /usr/local/bin
uv_source=
for source in /opt/dycon-wheels/uv /opt/dycon-skillsbench-tools/uv; do
    if test -x "$source"; then uv_source=$source; break; fi
done
if test -z "$uv_source"; then printf 'uv unavailable\n'; exit 0; fi
rm -f /root/.local/bin/uv /root/.local/bin/uvx /root/.local/bin/env
cp "$uv_source" /root/.local/bin/uv
printf '%s\n' '#!/bin/sh' 'exec /root/.local/bin/uv tool run "$@"' > /root/.local/bin/uvx
printf '%s\n' 'export PATH=/root/.local/bin:/opt/dycon-wheels:$PATH' > /root/.local/bin/env
chmod 755 /root/.local/bin/uv /root/.local/bin/uvx
chmod 644 /root/.local/bin/env
if test -x /opt/dycon-skillsbench-tools/curl; then
    rm -f /usr/local/bin/curl
    cp /opt/dycon-skillsbench-tools/curl /usr/local/bin/curl
    chmod 755 /usr/local/bin/curl
    sha256sum /usr/local/bin/curl
fi
sha256sum /root/.local/bin/uv /root/.local/bin/uvx /root/.local/bin/env
/root/.local/bin/uv --version
'''


async def prepare_verifier_tools(env):
    result = await env.exec(verifier_tool_command(), user='root', timeout_sec=30)
    if result.return_code:
        raise RuntimeError('Could not prepare trusted writable SkillsBench verifier tools')
    return result
