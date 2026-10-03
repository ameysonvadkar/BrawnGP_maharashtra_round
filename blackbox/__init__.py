"""Black Box: a flight recorder for AI agents.

Decorator API for recording your own agent (see blackbox/sdk.py):
    import blackbox as bb
    @bb.step("retrieve")
    def search(name): ...
"""
from blackbox.sdk import STEP_TYPES, get_agent, record, register_agent, state, step

__all__ = ["STEP_TYPES", "get_agent", "record", "register_agent", "state", "step"]
