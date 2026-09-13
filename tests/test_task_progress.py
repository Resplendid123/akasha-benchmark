import io
from contextlib import closing
from types import SimpleNamespace

import pytest

from akasha_benchmark import progress
from akasha_benchmark.store import connect, repo
from akasha_benchmark.store.migrate import migrate
from akasha_platform import tasks
from akasha_platform.settings import Settings


def test_nested_steps_map_to_overall_progress(capsys):
    with progress.scope(1, 2, 3, '归一化'):
        with progress.scope(0, 1, 2, 'hotpotqa'):
            progress.report(1, 2, '样本')
    events = [progress.parse(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]['done'] == 4167
    assert events[-1]['total'] == 10000
    assert '归一化 · hotpotqa · 样本' in events[-1]['note']
    progress.report(1, 2, '独立任务')
    assert progress.parse(capsys.readouterr().out.strip())['done'] == 5000


@pytest.mark.parametrize('line', [
    'ordinary log', 'AKASHA_PROGRESS nope', 'AKASHA_PROGRESS []',
    'AKASHA_PROGRESS {"done": 2, "total": 1, "note": "bad"}',
    'AKASHA_PROGRESS {"done": 0, "total": 0, "note": "bad"}',
])
def test_invalid_progress_is_ignored(line):
    assert progress.parse(line) is None


@pytest.mark.parametrize('exit_code, expected', [(0, 10000), (1, 5000)])
def test_task_progress_is_not_overwritten_by_legacy_logs(tmp_path, capsys, exit_code, expected):
    path = tmp_path / 't.db'
    migrate(path, verbose=False)
    with closing(connect(path)) as connection:
        task_id = repo.create_task(connection, stage='evaluate', argv=[])
        connection.commit()
    progress.report(1, 2, '指标计算')
    line = capsys.readouterr().out
    process = SimpleNamespace(stdout=io.StringIO(line + 'hotpotqa: 100/100\n'), wait=lambda: exit_code)
    tasks._pump(task_id, process, Settings(db_path=path), tmp_path / 'task.log')
    with closing(connect(path, read_only=True)) as connection:
        task = repo.get_task(connection, task_id)
    assert task['progress_done'] == expected
    assert task['progress_total'] == 10000
    assert task['status'] == ('succeeded' if exit_code == 0 else 'failed')


def test_failed_final_step_does_not_show_100_percent(tmp_path, capsys):
    path = tmp_path / 't.db'
    migrate(path, verbose=False)
    with closing(connect(path)) as connection:
        task_id = repo.create_task(connection, stage='download', argv=[])
        connection.commit()
    progress.report(1, 1, '校验完成')
    process = SimpleNamespace(stdout=io.StringIO(capsys.readouterr().out), wait=lambda: 1)
    tasks._pump(task_id, process, Settings(db_path=path), tmp_path / 'task.log')
    with closing(connect(path, read_only=True)) as connection:
        assert repo.get_task(connection, task_id)['progress_done'] == 9999


@pytest.mark.parametrize('output', ['', 'hotpotqa: 2/5\n'])
def test_successful_task_without_structured_progress_reaches_100(tmp_path, output):
    path = tmp_path / 't.db'
    migrate(path, verbose=False)
    with closing(connect(path)) as connection:
        task_id = repo.create_task(connection, stage='subset', argv=[])
        connection.commit()
    process = SimpleNamespace(stdout=io.StringIO(output), wait=lambda: 0)
    tasks._pump(task_id, process, Settings(db_path=path), tmp_path / 'task.log')
    with closing(connect(path, read_only=True)) as connection:
        task = repo.get_task(connection, task_id)
    assert task['status'] == 'succeeded'
    assert task['progress_done'] == task['progress_total'] == 10000
