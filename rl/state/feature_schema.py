from dataclasses import dataclass


@dataclass(frozen=True)
class StateFeatureSchema:
    """
    Machine-readable schema defining dimension specifications for the Phase 6 RL state vector.

    Dimensions:
    -----------
    objective_dim            : 64  (Text embedding of agent_intent string)
    candidate_text_dim       : 64  (Text embedding of canonical UICandidate string)
    candidate_structural_dim : 20  (Categorical tag/type encodings and boolean feature flags)
    candidate_visual_dim     : 7   (Normalized bounding box geometry & area ratio)
    candidate_feature_dim    : 91  (candidate_text_dim + candidate_structural_dim + candidate_visual_dim)
    max_candidates           : 20  (Fixed candidate capacity per observation)
    candidate_matrix_shape   : (20, 91)
    candidate_mask_shape     : (20,)
    context_dim              : 30  (Workflow one-hot [4], operation one-hot [3], step progress [1],
                                    previous public structural features [20], has_prev_action [1], prev_success [1])
    total_flat_dim           : 64 + (20 * 91) + 20 + 30 = 1934
    """

    objective_dim: int = 64
    candidate_text_dim: int = 64
    candidate_structural_dim: int = 20
    candidate_visual_dim: int = 7
    max_candidates: int = 20
    context_dim: int = 30
    encoding_version: str = "phase6-public-prev-candidate-v2"

    @property
    def candidate_feature_dim(self) -> int:
        return self.candidate_text_dim + self.candidate_structural_dim + self.candidate_visual_dim

    @property
    def candidate_matrix_shape(self) -> tuple[int, int]:
        return (self.max_candidates, self.candidate_feature_dim)

    @property
    def candidate_mask_shape(self) -> tuple[int]:
        return (self.max_candidates,)

    @property
    def total_flat_dim(self) -> int:
        return self.objective_dim + (self.max_candidates * self.candidate_feature_dim) + self.max_candidates + self.context_dim
