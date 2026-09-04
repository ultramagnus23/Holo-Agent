"""Domain plugin packages (master-build Milestone 3: separate HoloAgent's
generic core -- routing, orchestration, provenance, the tool contract and
registry, evaluation -- from domain-specific computational tools).

Each subpackage here is a self-contained plugin: it owns its own physics
backend, its own tool implementations, and registers those tools with
holoagent.tools.registry by being imported. Nothing in holoagent's core
(orchestrator/, provenance/, router/, eval/, tools/contract.py,
tools/registry.py) imports from here or knows these packages exist --
the dependency points one way, domain -> core, never core -> domain.

Currently one domain: holoforge (computational holography, wrapping the
frozen HoloForge Part 2 twin). A future general-optics domain would live
alongside it here as its own package, e.g. holoagent/domains/waveoptics/,
without touching anything under holoforge/ or core.
"""
