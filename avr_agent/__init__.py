"""AVR Agent — Adaptive Verify-and-Refine (Phase 1: Compute-Aware MVP).

Pipeline:  Router -> Generator -> Verifier -> Selector
"""
from .router import Router, Difficulty
from .generator import Generator, Candidate
from .verifier import Verifier, MathVerifier, CodeVerifierAdapter
from .selector import Selector, SelectionResult
from .agent import AVRAgent, AgentResult

__all__ = ["Router", "Difficulty", "Generator", "Candidate", "Verifier",
           "MathVerifier", "CodeVerifierAdapter", "Selector",
           "SelectionResult", "AVRAgent", "AgentResult"]
