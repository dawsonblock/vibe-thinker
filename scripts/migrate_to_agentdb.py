#!/usr/bin/env python3
"""Standalone CLI wrapper for AgentDB migration.

The importable logic lives in ``agentdb_migration.py`` so it is available
from wheel installs as well as source checkouts.
"""

from __future__ import annotations

import sys

from agentdb_migration import main

if __name__ == "__main__":
    sys.exit(main())
