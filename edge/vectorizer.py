import numpy as np

def to_vector(signal):
    fft = np.fft.fft(signal)

    features = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.sum(np.abs(fft))
    ]

    vec = np.array(features)
    return (vec / np.linalg.norm(vec)).tolist()