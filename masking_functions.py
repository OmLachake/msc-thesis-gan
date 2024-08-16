# Om Lachake, Aug 2024

import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_black_mask(height, width):
    """
    Creates a black mask of the given dimensions.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)

def create_random_noise_mask(height, width):
    """
    Creates a random noise mask of the given dimensions.
    """
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return noise

def create_gaussian_noise_mask(height, width, mean=85, std=25):
    """
    Creates a Gaussian noise mask of the given dimensions using Gaussian distribution.
    """
    noise = np.random.normal(mean, std, (height, width, 3))
    noise = np.clip(noise, 0, 255)
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    return noise.astype(np.uint8)

def create_voronoi_noise_mask(height, width, num_cells=20):
    """
    Creates a Voronoi noise mask of the given dimensions.
    """
    seed_points = np.random.randint(0, min(height, width), (num_cells, 2))
    colors = np.random.randint(0, 256, (num_cells, 3), dtype=np.uint8)
    voronoi_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            distances = np.sqrt((seed_points[:, 0] - y) ** 2 + (seed_points[:, 1] - x) ** 2)
            closest_seed = np.argmin(distances)
            voronoi_mask[y, x] = colors[closest_seed]
    
    return voronoi_mask

def create_perlin_noise_mask(height, width, scale=10):
    """
    Creates a Perlin noise mask of the given dimensions.
    """
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(t, a, b):
        return a + t * (b - a)

    def grad(hash, x, y):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 or h == 12 or h == 14 else x
        return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)

    def perlin(x, y):
        xi = int(x) & 255
        yi = int(y) & 255
        xf = x - int(x)
        yf = y - int(y)
        u = fade(xf)
        v = fade(yf)

        n00 = grad(p[perm[xi] + yi], xf, yf)
        n01 = grad(p[perm[xi] + yi + 1], xf, yf - 1)
        n10 = grad(p[perm[xi + 1] + yi], xf - 1, yf)
        n11 = grad(p[perm[xi + 1] + yi + 1], xf - 1, yf - 1)

        x1 = lerp(u, n00, n10)
        x2 = lerp(u, n01, n11)
        return (lerp(v, x1, x2) + 1) / 2

    global perm, p
    perm = np.arange(256, dtype=int)
    np.random.shuffle(perm)
    p = np.concatenate([perm, perm])

    perlin_noise = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            r = perlin(x / scale, y / scale)
            g = perlin((x + 100) / scale, (y + 100) / scale)
            b = perlin((x + 200) / scale, (y + 200) / scale)
            perlin_noise[y, x] = (
                int(r * 255),
                int(g * 255),
                int(b * 255)
            )

    return perlin_noise