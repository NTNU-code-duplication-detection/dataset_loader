"""
Module for smoothign graphs with different properties
"""
import scipy


def smooth_embeddings(embs, window=100, poly=3):
    """
    Smooth embedding grph at axis=1
    """
    # Apply smoothing across embedding dimension
    return scipy.signal.savgol_filter(embs, window, poly, axis=0)


def smooth_multiple_embeddings(embs_list, windows=100, poly=3):
    """
    Smooth multple embeddings
    """
    embs_list_smoothed = []
    for emb in embs_list:
        embs_list_smoothed.append(smooth_embeddings(emb, windows, poly))
    return embs_list_smoothed
