# SPDX-License-Identifier: Apache-2.0
"""Filter and report exact upstream pytest node IDs."""

import json
import os
from pathlib import Path

import pytest


def _emit(event):
    with Path(os.environ["SAIL_ASSET_EVENTS"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    selected = set(json.loads(os.environ["SAIL_ASSET_NODEIDS"]))
    kept = []
    dropped = []
    for item in items:
        if item.nodeid in selected:
            kept.append(item)
        else:
            dropped.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept
    _emit({"event": "collected", "nodeids": [item.nodeid for item in kept]})


def pytest_collectreport(report):
    if report.failed or report.skipped:
        _emit(
            {
                "event": "collection_report",
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "reason": report.longreprtext,
            }
        )


def pytest_runtest_logstart(nodeid, location):
    _emit({"event": "started", "nodeid": nodeid})


def pytest_runtest_logreport(report):
    reason = report.longreprtext if report.longrepr is not None else ""
    if report.failed:
        captured = "\n".join(content for _, content in report.sections)
        if captured:
            reason += "\n\nCaptured output (tail):\n" + captured[-12000:]
    _emit(
        {
            "event": "test_report",
            "nodeid": report.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
            "reason": reason,
            "xfail": getattr(report, "wasxfail", ""),
        }
    )
