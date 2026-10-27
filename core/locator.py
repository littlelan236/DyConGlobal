"""Path contract for run-local artifacts and external NAS tensors."""

from __future__ import annotations

from pathlib import Path

from core.io import identifier, owned_path, run_path


class RunLocator:
    def __init__(self, root, run_id):
        self.root = Path(root).resolve()
        self.run_id = identifier(run_id)
        self.run = run_path(self.root, self.run_id)

    def path(self, *parts):
        return owned_path(self.root, str(Path("runs") / self.run_id / Path(*parts)))

    @property
    def tasks(self):
        return self.path("tasks.json")

    @property
    def config(self):
        return self.path("config.yaml")

    @property
    def resolved_config(self):
        return self.path("resolved-config.json")

    def trajectory(self, rollout_id):
        return self.path("rollout", "trajectories", identifier(rollout_id), "raw.traj.json")

    def patch(self, rollout_id):
        return self.path("rollout", "patches", identifier(rollout_id), "patch.diff")

    def enrichment(self, *parts):
        return self.path("enrichment", *parts)

    def hidden(self, *parts):
        return self.path("hidden", *parts)

    def harness(self, *parts):
        return self.path("harness", *parts)

    def nas_hidden(self, nas_root, *parts):
        return Path(nas_root) / "hidden" / "online" / self.run_id / Path(*parts)
