import numpy as np 

def encode_ap_group(
    ap_group: list[int],
    all_ap_indices: np.ndarray,
) -> np.ndarray:
    """
    Convert a variable-length AP group containing actual network
    node indices into a fixed-size binary context vector.

    Example
    -------
    all_ap_indices = [0, 4, 8, 12, 16]
    ap_group       = [0, 8]

    returns:
        [1, 0, 1, 0, 0]
    """

    return np.isin(
        all_ap_indices,
        ap_group
    ).astype(np.float32)