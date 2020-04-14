import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename

from authlib.integrations.requests_client import OAuth2Session
import google.oauth2.credentials
import googleapiclient.discovery

import google_auth

UPLOAD_FOLDER = os.path.join('static','img')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'JPG'])

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['SECRET_KEY'] = os.environ.get("FN_FLASK_SECRET_KEY", default=False)

application.register_blueprint(google_auth.app)


@application.route('/', methods=['GET'])
def index(photo=None):
    import boto3
    import os

    #get the latest photo
    #download from s3
    sess = boto3.session.Session()
    s3 = sess.client(
            service_name='s3',
            region_name='us-east-2'
    )
    s3.download_file('woolly-bear', 'latest.txt', 'latest.txt')

    with open('latest.txt','r') as f:
        latest = f.read()

    s3.download_file('woolly-bear','photos/output/' + latest, os.path.join('static','img','latest',latest))
    photo = os.path.join('static','img','latest',latest)

    return render_template(
        'index.html', photo=photo)

@application.route('/how/', methods=['GET'])
def how(photo=None):
    import boto3
    import os

    #a page for how it works

    return render_template(
        'how.html')

@application.route('/upload/', methods=['GET', 'POST'])
def upload(photo=None, res=None):
    import model
    import output
    import urllib
    import json

    userdata=None
    if google_auth.is_logged_in():
        userdata = google_auth.get_user_info()  #I think its a dict
    else:
        return redirect('/google/login')

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
            flip = model.photoReshapei(os.path.join(application.config['UPLOAD_FOLDER'], filename))
#            print('Photo flip: {:}'.format(flip))
            photoProcess = os.path.join(application.config['UPLOAD_FOLDER'], 'Processed',filename)
            try:
                ip = request.environ.get('HTTP_X_FORWARDED_FOR') or request.environ.get('REMOTE_ADDR')
#                ip = '97.115.130.44'
                locurl = 'https://tools.keycdn.com/geo.json?host={:}'.format(ip)  #its more complicated than this
                response = urllib.request.urlopen(locurl)
                ipdata = json.load(response)['data']['geo']
                if not ipdata['city']:
                    ipdata = {}
            except:
                ipdata = {}

            iswb = output.screen(photoProcess, flip=flip)
            if iswb:
                res = output.predict(photoProcess, flip=flip, ipdata=ipdata)

            #go get the photo either way
            photo = os.path.join(application.config['UPLOAD_FOLDER'], 'Processed','output',filename)
            output.upload(iswb, fn = os.path.join(application.config['UPLOAD_FOLDER'], 'Processed',filename), ipdata=ipdata, userdata=userdata)


    return render_template('upload.html', photo=photo)


def main():
    application.run(debug=False)
#    application.run(ssl_context="adhoc")  #only for development with https

if __name__ == '__main__':
    main()
