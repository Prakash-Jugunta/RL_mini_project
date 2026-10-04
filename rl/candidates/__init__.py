# Phase 5: DOM Candidate Extraction Pipeline
# Import directly from the module (not the rl.env package) to avoid circular imports.
from rl.candidates.extractor import RealDOMCandidateExtractor, CandidateExtractionConfig

__all__ = [
    "RealDOMCandidateExtractor",
    "CandidateExtractionConfig",
]
