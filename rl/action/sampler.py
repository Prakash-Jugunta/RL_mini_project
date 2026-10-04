import numpy as np


def sample_valid_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    """
    Uniformly sample a random valid candidate index i from indices where mask[i] == 1.

    Parameters
    ----------
    mask : np.ndarray
        Action validity mask of shape (20,) with values 1 (valid) or 0 (padded).
    rng : np.random.Generator
        Seeded NumPy random number generator.

    Returns
    -------
    action_index : int
        A valid candidate index in range [0, 19] where mask[action_index] == 1.
        If all mask entries are 0 (empty candidate list), returns 0 safely.
    """
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) == 0:
        return 0
    return int(rng.choice(valid_indices))
