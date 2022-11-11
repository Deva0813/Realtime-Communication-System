from flask import Flask ,render_template,Response,jsonify
import opencv
import trainlist

video_camera = None
global_frame = None
app=Flask(__name__)

        
@app.route("/label")
def label_text():
    index=opencv.get_frame()[1]
    dataset=trainlist.dataset
    dataset.append("-")
    label=dataset[index]
    return jsonify(label)

@app.route("/")
def index():
    opencv.cap.release()
    return render_template('index.html')

@app.route("/translate")
def translate():
    opencv.cap=opencv.cv.VideoCapture(0)
    txt=label_text()
    return render_template('video_out.html',txt=txt.json)

def gen_vid():             # Video Stream
    global video_camera 
    global global_frame  
    while True:
        frame =opencv.get_frame()[0]

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
    
@app.route("/video")
def video():
    return Response(gen_vid(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    

@app.route("/about")
def about():
    opencv.cap.release()
    return render_template('about_us.html')

@app.route("/signup")
def sign_up():
    opencv.cap.release()
    return render_template('signup.html')

@app.route("/profile")
def profile():
    opencv.cap.release()
    return render_template('profile.html')

# Copyrigths:
# Devanand
# Dhinesh
# opencv.cap.release()


if __name__=="__main__":
    app.run(host='0.0.0.0', threaded=True , debug=True)