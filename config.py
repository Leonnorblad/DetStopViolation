import cv2
import supervision as sv
from inference import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from roboflow import Roboflow
import numpy as np
import os

ROAD = {"example":np.array([[6, 404],[552, 535],[1073, 611],[1103, 626],[830, 771],[3, 526]])}

STOP_ZONE = {"example":np.array([[500, 668],[767, 559],[1094, 604],[1103, 626],[833, 768],[500, 668]])}

OUT_ZONE = {"example":np.array([[1079, 601],[1194, 611],[1364, 632],[1664, 711],[1352, 920],[830, 771],[1094, 635],[1097, 620],[1079, 601]])}

POINT= {"example":(961, 665)}