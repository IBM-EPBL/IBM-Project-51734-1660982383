# define path to train and test dir

trainingpath=r"dataset/spiral/training"
testingpath=r"dataset/spiral/testing"

#loading train and test data

print("[INFO] loading data...")
(X_train,Y_train)=load_split(trainingpath)
(X_test,Y_test)=load_split(testingpath)