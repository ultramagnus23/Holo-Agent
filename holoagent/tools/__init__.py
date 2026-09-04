"""Core tool infrastructure: the typed contract (contract.py) and the
registry (registry.py) every domain plugin's tools register into. This
package itself is domain-agnostic -- it has no knowledge of HoloForge or
any other physics backend (master-build Milestone 3: separate HoloAgent's
core from domain-specific tools; see holoagent/domains/__init__.py).

Populating the registry with actual tools is a domain plugin's job, done
by importing it (e.g. `import holoagent.domains.holoforge`) -- that
import's side effect is every tool in that domain calling
@register(...). For backward compatibility, since HoloForge is currently
the only bundled domain and a lot of existing code (including most of
this project's own tests) still does `import holoagent.tools` expecting
the registry to already be populated, importing THIS package also loads
the holoforge domain by default. A second domain, once one exists, would
need its own explicit `import holoagent.domains.<name>` -- this package
will not guess which domains a caller wants; the default-load behavior
here is a one-domain-era convenience, not a general plugin-discovery
mechanism.
"""

from holoagent.tools.registry import get_tool, list_tools, register  # noqa: F401

import holoagent.domains.holoforge  # noqa: F401  (registration side effect -- see module docstring)
