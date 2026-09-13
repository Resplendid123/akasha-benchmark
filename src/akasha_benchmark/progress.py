"""结构化任务进度。嵌套步骤映射到整项任务的 0–10000 区间。"""

import json
from contextlib import contextmanager
from contextvars import ContextVar

PREFIX = 'AKASHA_PROGRESS '
_scope = ContextVar('progress_scope', default=(0.0, 1.0, ''))


def report(done: float, total: float, note: str) -> None:
    start, end, parent = _scope.get()
    fraction = max(0.0, min(1.0, done / total)) if total else 0.0
    label = ' · '.join(part for part in (parent, note) if part)
    print(PREFIX + json.dumps({
        'done': round((start + (end - start) * fraction) * 10000),
        'total': 10000,
        'note': f'{label}（{done:g}/{total:g}）',
    }, ensure_ascii=False), flush=True)


@contextmanager
def scope(start: float, end: float, total: float, label: str):
    """将内部进度映射到父步骤的指定范围；异常不会自动标记完成。"""
    parent_start, parent_end, parent_label = _scope.get()
    width = parent_end - parent_start
    token = _scope.set((
        parent_start + width * start / total,
        parent_start + width * end / total,
        ' · '.join(part for part in (parent_label, label) if part),
    ))
    try:
        report(0, 1, '开始')
        yield
    finally:
        _scope.reset(token)


def parse(line: str) -> dict | None:
    if not line.startswith(PREFIX):
        return None
    try:
        value = json.loads(line[len(PREFIX):])
        done, total = value['done'], value['total']
        if type(done) is not int or type(total) is not int or total <= 0 or not 0 <= done <= total:
            return None
        if not isinstance(value['note'], str):
            return None
        return value
    except (ValueError, KeyError, TypeError):
        return None
