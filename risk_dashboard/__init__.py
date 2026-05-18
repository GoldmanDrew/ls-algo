"""Risk dashboard package.

Reads outputs from ``ibkr_flex.py`` + ``ibkr_accounting.py`` (under
``data/runs/<RUN_DATE>/``) and produces a JSON snapshot consumed by the
static site under ``site/``.

The snapshot is committed into the private ``ls-algo`` repo on each EOD
run; the static site (deployed via GitHub Pages) fetches the snapshot at
runtime via the GitHub Contents API using a user-supplied PAT, so that
only collaborators on the private repo can see real numbers.

See ``risk_dashboard/README.md`` for the deployment + auth setup.
"""

from __future__ import annotations

__version__ = "0.1.0"
