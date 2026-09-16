import json
from pathlib import Path
from c2kv_eval.analysis import check_persistent_history_lifecycle as checker


def test_identity_lifecycle_is_skipped_after_completeness_check(tmp_path, monkeypatch, capsys):
    logs = tmp_path/'h2o'/'logs'
    logs.mkdir(parents=True)
    rows = [{"id": "a"}, {"id": "b"}]
    (logs/'details.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (logs/'summary.json').write_text(json.dumps({"history_kv_lifecycle_mode": "stateless_reselect"}))
    monkeypatch.setattr('sys.argv', ['check', '--run-root', str(tmp_path),
                                    '--expected-examples', '2', '--methods', 'h2o'])
    checker.main()
    assert 'SKIP h2o' in capsys.readouterr().out
