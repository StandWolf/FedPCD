# ---------------------------------------------------------------------------
# New in FedPCD (2026), licensed under the Apache License, Version 2.0.
# See NOTICE.
# ---------------------------------------------------------------------------
"""Client-level missing-rate helpers."""


def compute_client_missing_rate(sim_entries):
    """Fraction of a client's samples with at least one missing modality.

    ``sim_entries`` is the list returned by
    ``DataloadManager.get_client_sim_dict``. Each entry ends with a mask list
    whose first two elements are the accelerometer and gyroscope missing flags
    (1 = missing, 0 = present).
    """
    if not sim_entries:
        return 0.0
    missing = sum(1 for entry in sim_entries
                  if entry[-1][0] == 1 or entry[-1][1] == 1)
    return missing / len(sim_entries)
