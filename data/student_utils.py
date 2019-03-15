import numpy as np

class StampedData():
    def __init__(self):
        self.data = []
        self.t = []

    def convert_lists_to_numpy(self):
        self.data = np.array(self.data)
        self.t = np.array(self.t)