import base64
import io
import shutil
import uuid
import cv2
import os
import numpy as np
import json
from PIL import Image
from flask import Flask, request, Response, render_template

from object_detect import detectImage

image_path = 'data/images/'

app = Flask(__name__)


@app.route('/createFolder')
def create_folder():
    folder_name = str(uuid.uuid1())
    os.mkdir(image_path + folder_name)
    return Response(response=folder_name, status=200, mimetype='application')


@app.route('/deleteFolder')
def delete_folder():
    if request.args.get('folder_name'):
        folder_name = str(request.args.get('folder_name'))
        if os.path.exists(image_path + folder_name):
            shutil.rmtree(image_path + folder_name)
        return Response(response='done', status=200, mimetype='application')


@app.route('/detect')
def detect():
    # folder_name = image_path
    if request.args.get('folder_name'):
        folder_name = image_path + str(request.args.get('folder_name')) + '/'

        # video_name = 'result.avi'
        images = list()
        file_names = os.listdir(folder_name)
        file_names.sort()
        # i = 1
        for file_name in file_names:
            file_path = os.path.join(folder_name, file_name)
            file_size = os.path.getsize(file_path)
            if file_size != 0:
                data = dict()

                frame = cv2.imread(file_path)
                result = detectImage(frame)
                frame = cv2.flip(result, 1)
                imgencode = cv2.imencode('.jpg', frame)[1]
                # base64 encode
                stringData = base64.b64encode(imgencode).decode('utf-8')
                b64_src = 'data:image/jpg;base64,'
                stringData = b64_src + stringData

                data['data'] = stringData
                images.append(data)

        final_data = json.dumps({'files': images}, sort_keys=True, indent=4, separators=(',', ': '))
        return Response(response=final_data, status=200, mimetype='application')


@app.route('/sendImage', methods=['POST'])
def send_image():
    folder_name = image_path
    video_frame = ''
    if request.args.get('folder_name'):
        folder_name = image_path + str(request.args.get('folder_name')) + '/'

    if request.args.get('video_frame'):
        video_frame = str(request.args.get('video_frame'))

    if request.form['image']:
        file = request.form.get('image')
        b = io.BytesIO(base64.b64decode(file))
        pimg = Image.open(b)
        frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
        filename = video_frame + '_' + str(uuid.uuid1())
        cv2.imwrite(folder_name + str(filename) + '.jpg', frame)

        return Response(response='upload ok', status=200, mimetype='application')


if __name__ == '__main__':
    app.run(debug=True)
