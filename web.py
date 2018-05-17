import os
import time
from flask import Flask, request, redirect, url_for
from detector import Detector

UPLOAD_FOLDER = 'uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

det = Detector()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(filepath)
        res = det.detect(filepath)
        print(res)
        if res is not None:
            return res
        else:
            return 'nope'
    return '''
    <!doctype html>
    <title>Upload</title>
    <h1>Upload image</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

app.run(host= '0.0.0.0', port=5353, threaded=False)
