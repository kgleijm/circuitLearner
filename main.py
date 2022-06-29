import os
from typing import Callable
from videophy import Stream
from PIL import Image, ImageDraw
import keyboard
from GeneticAlgorithm import GeneticAlgorithm


# choose
# observation: distance sensor (4-8* percentage of max distance) = [sens1, sens2, sens3, sens4]
# options: -1|+1(up, right, down, left) nothing
# possibilities: whether you're at max or min speed

class SimpleMatrix:

    def __init__(self, twoDArarray: list[list[int]], tilesize=10):
        self.tilesize = tilesize
        self._data = twoDArarray
        self.width = len(twoDArarray[0])
        self.height = len(twoDArarray)

        self.img = self.render()

    def render(self):
        img = Image.new("RGB", (self.width, self.height))
        for y in range(self.height):
            for x in range(self.width):
                img.putpixel((x, y), (255, 255, 255) if not self[x, y] else (0, 0, 0))
        return img

    def __getitem__(self, item):
        x, y = item
        return self._data[int(y)][int(x)]

    def __repr__(self):
        outPath = ''
        for row in self._data:
            outPath += "["
            for e in row:
                outPath += f"{e}, "
            outPath += "]"
        return outPath

    def getRendering(self, dot=(-1, -1), zones=False, score=None):

        img = self.img.copy()

        # draw Zones
        draw = ImageDraw.Draw(img)
        if dot != (-1, -1):
            draw.ellipse([dot[0] - 3, dot[1] - 3, dot[0] + 3, dot[1] + 3], fill=(255, 0, 0))

        if score is not None:
            draw.text((10, 10), f"score: {score}", fill=(0, 0, 0))

        return img

    def ifPixel(self, x, y):
        return self[x, y]


class Helpers:

    @staticmethod
    def showMatrix(mat: SimpleMatrix):
        pass


class Sim:

    def __init__(self, onTick: Callable):

        """
        onTick applies xSpeed and ySpeed to position

        :param onTick: takes a function that changes xSpeed or ySpeed
        """

        # Open the image form working directory
        self.onTick = onTick
        self.imagePath = os.path.join("circuits", "1127809196775976634.png")
        self.image: Image = Image.open(self.imagePath)

        self.maxSpeed = 3

        self.xSpeed = 0
        self.ySpeed = 0
        self.xPos = 450
        self.yPos = 550

        self.score = 0

        # convert pixels to matrix
        self.twoD = []
        for y in range(self.image.height):
            row = []
            for x in range(self.image.width):
                val = 0 if int(sum(self.image.getpixel((x, y))) / 3) > 128 else 1
                row.append(val)
            self.twoD.append(row)
        self.matrix = SimpleMatrix(self.twoD)

    def getTileCoords(self, tileSize=100):
        return int(self.xPos // tileSize), int(self.yPos // tileSize)

    def reset(self):
        self.xSpeed = 0
        self.ySpeed = 0
        self.xPos = 450
        self.yPos = 550

        self.score = 0

    def start(self, show=True):

        stream = Stream()
        lastTiles = []
        lastTiles.append(self.getTileCoords())

        while True:

            self.onTick(self)

            self.xPos += self.xSpeed
            self.yPos += self.ySpeed

            if show:
                rendering = self.matrix.getRendering(dot=(self.xPos, self.yPos), score=self.score)
                stream.newFrame(rendering)

            # new tile entered
            coords = self.getTileCoords()
            if coords != lastTiles[-1]:
                print(self.getTileCoords())

                if coords in lastTiles[:-1]:
                    print("-1")
                    self.score -= 1
                else:
                    print("+1")
                    self.score += 1

                lastTiles.append(self.getTileCoords())



            if len(lastTiles) >= 5:
                lastTiles.pop(0)

            # Check whether or not state is faulty
            if self.matrix.ifPixel(self.xPos, self.yPos):
                return self.score


def checkManualInput(sim: Sim):
    if keyboard.is_pressed('a'):
        sim.xSpeed -= 0.2 if sim.xSpeed > -sim.maxSpeed else 0
    if keyboard.is_pressed('d'):
        sim.xSpeed += 0.2 if sim.xSpeed < sim.maxSpeed else 0
    if keyboard.is_pressed('w'):
        sim.ySpeed -= 0.2 if sim.ySpeed > -sim.maxSpeed else 0
    if keyboard.is_pressed('s'):
        sim.ySpeed += 0.2 if sim.ySpeed < sim.maxSpeed else 0


sim = Sim(checkManualInput)

while True:
    sim.start(show=True)
    sim.reset()

# Simulation(None, None)
