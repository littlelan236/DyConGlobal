"""Full-policy GDN identity checks; metadata validation stays Torch-free."""


def checkpoint_identity(contract):
    identity = contract.get("gdn_execution")
    configured = contract["schedule"]["model"].get("gdn_backend", "reference_pytorch")
    if configured == "fla_chunk" and identity is None:
        raise ValueError("Full FLA checkpoint is missing its GDN execution identity")
    if identity is not None and (identity.get("configured_backend") != configured
                                 or identity.get("resolved_backend") != configured):
        raise ValueError("Full checkpoint GDN backend identity mismatch")
    initial = contract.get("initial_policy", {})
    if configured == "fla_chunk" and initial.get("gdn_execution") != identity:
        raise ValueError("Full FLA anchor GDN execution identity mismatch")
    return identity


def bind(model, context, expected=None):
    from RL.gdn_backend import install_gdn_backend
    binding = install_gdn_backend(model, expected_identity=expected)
    identity = binding.execution_contract()
    if any(peer != identity for peer in context.gather(identity)):
        raise ValueError("Full workers disagree on GDN execution identity")
    return binding


def evidence(binding):
    return {"execution": binding.execution_contract(), "observations": binding.observations()}
