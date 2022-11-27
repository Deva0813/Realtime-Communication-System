from flask import Flask ,render_template,Response,jsonify,request
import opencv
import trainlist
import webbrowser
import backend.mongo as mongo

video_camera = None
global_frame = None

app=Flask(__name__)

        
@app.route("/label")  #for label
def label_text():
    index=opencv.get_frame()[1]
    dataset=trainlist.dataset
    dataset.append("-")
    label=dataset[index]
    return jsonify(label)

@app.route("/home") #for home page
def index():
    opencv.cap.release()
    return render_template('index.html')

@app.route("/translate") #for translation
def translate():
    opencv.cap=opencv.cv.VideoCapture(0)
    txt=label_text()
    return render_template('video_out.html',txt=txt.json)
    
def gen_vid():     # Video Stream
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
    
@app.route("/video")    # Video page
def video():
    return Response(gen_vid(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/about")    # About page
def about():
    opencv.cap.release()
    return render_template('about_us.html')


@app.route("/signup")   # Sign up page
def sign_up():
    opencv.cap.release()
    return render_template('sign_up.html')

@app.route("/") #for login page
def login():
    opencv.cap.release()
    return render_template('login.html',userinfo=None)

@app.route("/profile")      # Profile page
def profile():
    opencv.cap.release()
    return render_template('profile.html')

@app.route("/choice")       # Choice page
def choice():
    opencv.cap.release()
    return render_template('choice.html')

@app.route("/audio")        # Audio page
def audio():
    opencv.cap.release()
    return render_template('audio_out.html')

@app.route('/validate', methods=['POST'])
def validate_sign():
    email = request.form['email']
    password = request.form['password']
    if mongo.validate(email,password):
        userinfo=mongo.show(email)
        return render_template('login.html',accept="success",userinfo=userinfo,email=email,password=password)
    else:
        return render_template('login.html',accept="failed",userinfo=None,email=email,password=None)
    
@app.route('/signup', methods=['POST'])
def getvalue():
    username = request.form['name']
    email = request.form['email']
    password = request.form['password']
    disability = request.form['inputDisability']
    role = request.form['inputRole']
    if username and email and password and disability and role:
        if mongo.insert(username,email,password,disability,role):
            accept="success"
            return render_template('sign_up.html',accept=accept,email=email,username=username)
        else:
            accept="exist"
            return render_template('sign_up.html',accept=accept,email=email,username=username)
    else:
        accept="failed"
        return render_template('sign_up.html',accept=accept,email=email,username=username)
    


# Copyrigths:
# Devanand
# Dhinesh
# webbrowser.open("http://127.0.0.1:5000/")


if __name__=="__main__":
    app.run(host='0.0.0.0', threaded=True ,port=5000,debug=True)