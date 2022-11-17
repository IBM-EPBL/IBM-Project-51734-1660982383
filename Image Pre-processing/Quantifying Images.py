def quantify_image(image):
    #compute histogram of oriented gradients feature vector for the input image
    features=feature.hog(image,orientations=9,pixels_per_cell=(10,10),cells_per_block=(2,2),transform_sqrt=True,block_norm="L1")
    return features

def load_split(path):
    #grab list of images in the input dir,then initialize the list of data and class labels
    
    imagepaths=list(paths.list_images(path))
    data,labels=[],[]
    
    #loop over the image path
    for imagepath in imagepaths:
        #extract the class label from the filename
        label=imagepath.split(os.path.sep)[-2]
        
        #load the input image
        image=cv2.imread(imagepath)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image=cv2.resize(image,(200,200))
        image=cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        #quantify the image
        features=quantify_image(image)
        
        #update the data and labels
        data.append(features)
        labels.append(label)
        
    return (np.array(data),np.array(labels))