"""Container error contracts shared by worker environments."""


class EnvironmentSetupFailed(RuntimeError):
    def __init__(self, message, *, dependency_failure=None):
        super().__init__(message)
        self.dependency_failure = dependency_failure
