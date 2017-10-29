import pybullet as p

from tqdm import trange


class Camera:
    def __init__(self, width = 1920, height = 1080):
        self.width = width
        self.height = height
        self.cameras = []

    def add(self, *args, **kwargs):
        self.cameras.append((args, kwargs))

    def render(self, steps = 1, verbose = False):
        if len(self.cameras) == 0:
            self.cameras.append(([], {}))

        images = [[] for _ in self.cameras]
        depths = [[] for _ in self.cameras]
        masks = [[] for _ in self.cameras]
        for _ in trange(steps) if verbose else range(steps):
            for i, (args, kwargs) in enumerate(self.cameras):
                _, _, image, depth, mask = p.getCameraImage(self.width, self.height, *args, **kwargs)
                images[i].append(image)
                depths[i].append(depth)
                masks[i].append(mask)
            p.stepSimulation()

        if len(self.cameras) == 1:
            return images[0], depths[0], masks[0]
        else:
            return images, depths, masks
