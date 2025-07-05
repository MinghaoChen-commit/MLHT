import os
import pandas as pd
import cv2
import numpy as np
from natsort import natsorted
import glob
from pathlib import Path

from copy import deepcopy
import itertools


import networkx as nx
import pulp

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import os
from shutil import rmtree

import numpy as np

from scipy.spatial.distance import cdist

import uuid
import hashlib
import random

import cv2


from tools.DataStructureClass import Trajectory,  Tracklet, FramePairs, TrajectoryTracklet, VisualizeStructure

from tools.PrepareData import ReadData, ReadMOTData, txt_to_csv, ReadGMOTdata, PrepareGMOTdata, config
from tools.utils import GetBranchID, VisID, GetSimilarityMatrix, visualize, visualizeGMOT, SavetheResult, images_to_video, VideoBuilder, regenerateGraphwithFrame, regenerateGraphwithForest, Custom_Euclidean_Match, VisSet
from collections import defaultdict


from scipy.optimize import linprog
from itertools import combinations


from tools.DataStructureClass import Target , Trajectory,  Nstr

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import math
import cv2
import numba 
from numba import jit
import matplotlib.pyplot as plt 
from Graphmatch.Entity import Item
from Graphmatch.utilsformatch import *
from Graphmatch.Estimate import *
from Graphmatch.Calculate import *
from scipy.optimize import linear_sum_assignment
from Graphmatch.Match import *

from multiprocessing import Process
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import time
import torch

from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from AFLink.AppFreeLink import AFLink

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pickle