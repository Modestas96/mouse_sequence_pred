import numpy as np
import math
class MathUtils:
    @staticmethod
    def distance(x, y):
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
    @staticmethod
    def getAngle(x, y, relativePoint):
        xx = np.copy(x)
        yy = np.copy(y)
        xx -= relativePoint
        yy -= relativePoint
        ff, ss = yy[1] - xx[1], yy[0] - xx[0]
        return math.atan2(yy[1] - xx[1], yy[0] - xx[0])
    @classmethod
    def strecthLines(self, rez, param):
        newPath = np.copy(rez)
        for i in range(1, len(rez)):
            diffX = rez[i][0] - rez[i-1][0]
            diffY = rez[i][1] - rez[i-1][1]

            newPath[i][0] = newPath[i-1][0] + diffX * param
            newPath[i][1] = newPath[i - 1][1] + diffY * param
        return newPath
    @classmethod
    def rotatePoint(self, pnt, angle):
        pntcopy = np.copy(pnt)
        pntcopy[0] = math.cos(angle) * (pnt[0]) - math.sin(angle) * (pnt[1])
        pntcopy[1] = math.sin(angle) * (pnt[0]) + math.cos(angle) * (pnt[1])
        return pntcopy
    @classmethod
    def rotateLines(self, rez, angle):
        newPath = np.copy(rez)
        for i in range(1, len(rez)):
            rez[i] -= rez[0]
            newPath[i] = self.rotatePoint(rez[i], angle)
            rez[i] += rez[0]
            newPath[i] += rez[0]
        return newPath