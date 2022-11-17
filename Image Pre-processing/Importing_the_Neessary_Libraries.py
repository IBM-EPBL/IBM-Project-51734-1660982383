#import the pakages
import os.path
import pickle
import cv2
import numpy as np
from imutils import build_montages, paths
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
from skimage import feature