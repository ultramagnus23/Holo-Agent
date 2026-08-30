from holoagent.physics.holoforge_adapter import HoloForgeAdapter  # noqa: F401

_default_adapter: HoloForgeAdapter | None = None


def get_default_adapter() -> HoloForgeAdapter:
    """Lazily constructed singleton so importing holoagent.tools doesn't
    require a HoloForge checkout to exist until a tool is actually called.
    """
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = HoloForgeAdapter()
    return _default_adapter
