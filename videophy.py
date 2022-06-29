# import needed libraries
import os
import numpy


try:
    from cv2 import cv2
    from PIL import Image, ImageDraw, ImageColor
    import matplotlib.pyplot as plt
except:
    # install open-cv if not yet installed
    print("Installing necessary packages")
    os.system("pip install opencv-python")
    os.system("pip install pillow")
    os.system("pip install matplotlib")
    from cv2 import cv2
    from PIL import Image, ImageDraw, ImageColor
    import matplotlib.pyplot as plt


class Video:

    codecs = {
        "mp4": 'mp4v',
        "avi": 'I420'
    }

    def __init__(self, outputFolder, name="video", fps=30.0, outputType="mp4"):
        self.frameList: list[Image] = []
        self.outputPath: str = ""
        self.width = -1
        self.height = -1
        self.fps = fps
        self.outputType = outputType
        self.videoWriter = None
        self.name = name
        self.setOutputFolder(outputFolder)

    def setOutputFolder(self, path: str):
        try:
            os.mkdir(path)
        except:
            print("Unable to create output path, might already exist")
        self.outputPath = path

    def getFileName(self):
        return os.path.join(self.outputPath, f"{self.name}.{self.outputType}")

    def addFrame(self, frame: Image):

        if self.videoWriter is None:
            self.width, self.height = frame.size
            fourcc = cv2.VideoWriter_fourcc(*Video.codecs[self.outputType])
            self.videoWriter = cv2.VideoWriter(self.getFileName(), fourcc, self.fps, (self.width, self.height))
        self.videoWriter.write(numpy.array(frame))

    def release(self):
        cv2.destroyAllWindows()
        self.videoWriter.release()
        print("Released video to: " + self.getFileName())


class Stream:

    def __init__(self):
        pass

    def newFrame(self, frame):
        cv2.imshow("Stream", numpy.array(frame))
        cv2.waitKey(3)


