import json
import runpy
from contextlib import closing
from pathlib import Path

import pytest

from akasha_benchmark.datasets.hotpotqa import HotpotQAAdapter
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate
from akasha_platform import prepare_normalize
from akasha_platform.download import SCRIPTS


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(HotpotQAAdapter, 'expected_qa_rows', lambda self: 1)
    (tmp_path / 'hotpotqa.json').write_text(json.dumps([{
        '_id': 'one', 'question': 'question', 'answer': 'answer',
        'supporting_facts': [['Title', 0]],
    }]))
    (tmp_path / 'hotpotqa_corpus.json').write_text(json.dumps([
        {'idx': 1, 'title': 'Title', 'text': 'body'},
    ]))
    path = tmp_path / 'test.db'
    migrate(path, verbose=False)
    return ['--db', str(path), '--dataset-dir', str(tmp_path), '--dataset', 'hotpotqa']


def test_normalization_runs_validation_on_selected_dataset(inputs, capsys):
    assert prepare_normalize.main(inputs) == 0
    output = capsys.readouterr().out
    assert '原始文件校验通过' in output
    assert 'all datasets pass' in output
    with closing(connect(Path(inputs[1]), read_only=True)) as connection:
        assert [r['name'] for r in repo.list_datasets(connection)] == ['hotpotqa']


def test_invalid_source_prevents_normalization(inputs, tmp_path):
    (tmp_path / 'hotpotqa_corpus.json').write_text('{}')
    with pytest.raises(ValueError, match='non-empty JSON array'):
        prepare_normalize.main(inputs)
    with closing(connect(Path(inputs[1]), read_only=True)) as connection:
        assert repo.list_datasets(connection) == []


def test_download_rejects_invalid_json_structure(tmp_path):
    script = runpy.run_path(str(SCRIPTS / 'download_datasets.py'))
    path = tmp_path / 'invalid.json'
    path.write_text('{}')
    assert script['summarize']([path]) == 1


def test_post_normalization_validation_failure_fails_task(inputs, monkeypatch):
    monkeypatch.setattr(prepare_normalize.runpy, 'run_path', lambda path: {'main': lambda args: 1})
    assert prepare_normalize.main(inputs) == 1
