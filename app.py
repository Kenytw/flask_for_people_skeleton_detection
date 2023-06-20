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
from object_3d_detect import detect3dFirstImage, detect3dImage
from object_detect import detectImage

image_path = 'data/images/'

app = Flask(__name__)


@app.route('/api/createFolder')
def create_folder():
    folder_name = str(uuid.uuid1())
    os.mkdir(image_path + folder_name)
    return Response(response=folder_name, status=200, mimetype='application')


@app.route('/api/deleteFolder')
def delete_folder():
    if request.args.get('folder_name'):
        folder_name = str(request.args.get('folder_name'))
        if os.path.exists(image_path + folder_name):
            shutil.rmtree(image_path + folder_name)
        return Response(response='done', status=200, mimetype='application')


@app.route('/api/show3d')
def show3d():
    # folder_name = image_path
    if request.args.get('folder_name'):
        folder_name = image_path + str(request.args.get('folder_name')) + '/'
        file_names = os.listdir(folder_name)
        file_names.sort()

        result_array = []
        i = 1
        for file_name in file_names:
            file_path = os.path.join(folder_name, file_name)
            file_size = os.path.getsize(file_path)
            if file_size != 0:
                frame = cv2.imread(file_path)
                result = detect3dFirstImage(frame)
                cropped = [result[0], result[1]]
                if result[0] != (0, 0) and result[1] != (0, 0):
                    result_array.append(detect3dImage(frame, cropped))
                i += 1

        final_json = '{"result":['
        j = 1
        for result in result_array:
            if result[5] == 'good':
                if j > 1: final_json = final_json + ','

                img_width = result[2][0] / result[3]
                img_height = result[2][1] / result[3]
                scale_percent = img_width / 4
                img_width = img_width / scale_percent
                img_height = img_height / scale_percent
                img_z = img_width * 0.2

                landmark = result[1]

                center_x = round((((landmark.landmark[23].x * img_width) +
                                   (landmark.landmark[24].x * img_width)) / 2), 2)
                center_y = round((((landmark.landmark[23].y * img_height) +
                                   (landmark.landmark[24].y * img_height)) / 2), 2)
                center_z = round((((landmark.landmark[23].z * img_width) +
                                   (landmark.landmark[24].z * img_z)) / 2), 2)

                i = 0
                coo = '{'
                while i <= 32:
                    coo = coo + '"landmark' + str(i) + '": {"x": ' + str(
                        (landmark.landmark[i].x * img_width) - center_x) + ',"y": ' + str(
                        (landmark.landmark[i].y * img_height) - center_y) + ',"z": ' + str(
                        (landmark.landmark[i].z * img_z) - center_z) + '}'
                    if i < 32:
                        coo = coo + ','

                    i += 1

                coo = coo + '}'
                final_json = final_json + coo
                j += 1

        final_json = final_json + ']}'
        final_data = json.dumps(json.loads(final_json))
        return Response(response=final_data, status=200, mimetype='application')


@app.route('/api/detect')
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

                """if not os.path.exists(folder_name + '/result'):
                    os.mkdir(folder_name + '/result')

                cv2.imwrite(folder_name + '/result/' + str(i) + '_' + file_name, frame)
                i += 1"""

        final_data = json.dumps({'files': images}, sort_keys=True, indent=4, separators=(',', ': '))
        # return render_template("Images.html", final_data=final_data)
        return Response(response=final_data, status=200, mimetype='application')


@app.route('/api/sendImage', methods=['POST'])
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
