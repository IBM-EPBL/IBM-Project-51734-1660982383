from app import *
import os
from utils.predict import predict

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')



@app.route("/signin", methods=["POST", "GET"])
def signin():
    if request.method=="GET":
        return render_template('signin.html')


@app.route("/signup", methods=["POST", "GET"])
def signup():
    if request.method=="GET":
        return render_template('signup.html')


@app.route("/upload", methods=["POST", "GET"])
def upload():
    if request.method=="GET":      
        return render_template('upload.html')
    if request.method=="POST":
        imgType = request.form["type"]
        file = request.files['file']

        imgPath = os.path.join(app.config['UPLOAD_FOLDER'], "upImage.jpg")
        file.save(imgPath)
        print(imgPath)
        prediction = predict(imgPath, imgType)
        return render_template('result.html', prediction=prediction, path='/uploads/upImage.jpg')

if __name__ == "__main__":
    app.run(debug=False)