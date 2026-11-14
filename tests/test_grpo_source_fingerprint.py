"""Legacy source hashes remain auditable without blocking exact resume."""

from RL.artifacts import resume_contract_matches


def test_resume_ignores_source_fingerprint_but_checks_training_contract():
    old={"source_fingerprint":"old","schedule":{"seed":1},"input":{"dataset":"same"}}
    current={"source_fingerprint":"new","schedule":{"seed":1},"input":{"dataset":"same"}}
    assert resume_contract_matches(old,current)
    current["schedule"]={"seed":2}
    assert not resume_contract_matches(old,current)


def test_resume_accepts_new_contract_without_source_fingerprint():
    old={"source_fingerprint":"old","schedule":{"seed":1}}
    assert resume_contract_matches(old,{"schedule":{"seed":1}})
