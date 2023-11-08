import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


# def circle_dataset(n=8000):
#     rng = np.random.default_rng(42)
#     x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
#     y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
#     norm = np.sqrt(x**2 + y**2) + 1e-10
#     x /= norm
#     y /= norm
#     theta = 2 * np.pi * rng.uniform(0, 1, n)
#     r = rng.uniform(0, 0.03, n)
#     x += r * np.cos(theta)
#     y += r * np.sin(theta)
#     X = np.stack((x, y), axis=1)
#     X *= 3
#     return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    
    # Generate circle points
    x = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    y = np.round(rng.uniform(-0.5, 0.5, n)/2, 1)*2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm

    # Add noise
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)

    # Eyes: adding points for two small circles
    for eye_x in [-0.5, 0.5]: # x-coordinates for left and right eyes
        eye_y = 0.5  # y-coordinate (same for both eyes)
        eye_radius = 0.2
        t = 2 * np.pi * rng.uniform(0, 1, n//20)  # divide by 20 to have fewer points for eyes
        eye_points_x = eye_x + eye_radius * np.cos(t)
        eye_points_y = eye_y + eye_radius * np.sin(t)
        x = np.concatenate([x, eye_points_x])
        y = np.concatenate([y, eye_points_y])

    # Mouth: adding points for a semi-circle
    mouth_radius = 0.5
    t = np.pi * rng.uniform(0, 1, n//10)  # divide by 10 to have fewer points for mouth
    mouth_points_x = mouth_radius * np.cos(t)
    mouth_points_y = -(0.5 + mouth_radius * np.sin(t) - mouth_radius)  # adjust y to position the mouth correctly
    x = np.concatenate([x, mouth_points_x])
    y = np.concatenate([y, mouth_points_y])

    X = np.stack((x, y), axis=1)
    X *= 3
    
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def dino_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def get_dataset(name, n=8000):
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    else:
        raise ValueError(f"Unknown dataset: {name}")
