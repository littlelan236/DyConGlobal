import pytest

from application.execution.config_revision import revise_config
from application.execution.run_plan import load_plan
from core.io import sha256
from tests.test_capture_budget_revision import make_run


@pytest.mark.parametrize('flag', [True, False])
def test_identity_check_flag_is_revisable(cfg, flag):
    run = make_run(cfg)
    revise_config(cfg['project_root'], run.name, revision_id='identity',
                  updates={'model.verify_identity': flag}, reason='fixture')
    assert load_plan(cfg['project_root'], run.name)['config']['model']['verify_identity'] is flag


@pytest.mark.parametrize('flag', ['false', 0, 1, None])
def test_non_boolean_rejected(cfg, flag):
    run = make_run(cfg)
    before = sha256(run / 'resolved-config.json')
    with pytest.raises(ValueError):
        revise_config(cfg['project_root'], run.name, revision_id='identity',
                      updates={'model.verify_identity': flag}, reason='fixture')
    assert sha256(run / 'resolved-config.json') == before
