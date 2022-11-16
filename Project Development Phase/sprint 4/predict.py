import pickle
import cv2
from skimage import feature 

waveModel = pickle.load(open("ML/models/spiralModel.pkl", "rb"))
spiralModel = pickle.load(open("ML/models/spiralModel.pkl", "rb"))

code2cat = {0:"healthy", 1:"parkinson"}


def processImage(imagePath):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image = cv2.resize(image, (200, 200))
    image = cv2.threshold (image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) [1]
    return image


def quantify_image(image):
  features = feature.hog(image, orientations=9, pixels_per_cell=(13, 13), cells_per_block=(4, 4), transform_sqrt=True, block_norm="L1")
  return features


def predict(imgPath, modelType):
    img = processImage(imgPath)
    imgFeatures = quantify_image(img)

    if modelType=="spiral":
      pre = spiralModel.predict([imgFeatures])

    if modelType=="wave":
      pre = spiralModel.predict([imgFeatures])

    return code2cat[pre[0]]