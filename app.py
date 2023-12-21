"""
The web service provider.

Author: WANG Xiangzhi
Date: Dec-17-2023
"""
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid, base64, queue, time
import data_analysis2

_img_buffer_path = "img_buffer"
_img_buffer_list = []
_img_error__list = []
_process_queue = queue.Queue(maxsize=10)
_image_extension_white_list = ['jpg', 'jpeg', 'png', 'blob']

app = Flask(__name__, template_folder='www', static_folder='www', static_url_path='')
os.makedirs(_img_buffer_path, exist_ok=True)

def _file_extension_checker(filename):
    """
    Check the file extension.
    """
    return filename == _image_extension_white_list[-1] or ('.' in filename and \
           filename.split('.')[-1].lower() in _image_extension_white_list)


def _json_response_maker(result, data, code=200):
    """
    Return all the POST request in JSON format.
    """
    return jsonify({"result":result, "data":data}), code


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method != "POST":
        return _json_response_maker("error", "Bad request type.", 400)
    
    if 'image' not in request.files or 'mask' not in request.files:
        return _json_response_maker("error", "No image file uploaded.")
    
    image_file = request.files['image']
    mask_file = request.files['mask']
    filename = secure_filename(image_file.filename)

    if not _file_extension_checker(filename):
        return _json_response_maker("error", "Only .jpg or .png file is allowed.")
    
    if _process_queue.full():
        return _json_response_maker("error", "Too many requests.")
    
    id = "img_" + str(uuid.uuid4())

    try:
        os.makedirs(os.path.join(_img_buffer_path, id), exist_ok=True)
        image_file.save(os.path.join(_img_buffer_path, id, "img.png"))
        mask_file.save(os.path.join(_img_buffer_path, id, "mask.png"))
    except:
        return _json_response_maker("error", "Invalid file format.")

    _process_queue.put(id)
    _img_buffer_list.append(id)
    return _json_response_maker("success", id)


@app.route('/result', methods=['POST'])
def result():
    if request.method != "POST":
        return _json_response_maker("error", "Bad request type.", 400)
    
    id = request.form.get('id')
    if id is None:
        return _json_response_maker("error", "Bad request type.", 400)
    
    img1_path = os.path.join(_img_buffer_path, id, "inpainted1.png")
    img2_path = os.path.join(_img_buffer_path, id, "inpainted2.png")
    img3_path = os.path.join(_img_buffer_path, id, "inpainted3.png")

    if id in _img_error__list:
        return _json_response_maker("error", "Got an error while processing.", 500)

    if id in _img_buffer_list:
        return _json_response_maker("processing", "", 200)

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return _json_response_maker("processing", "")
    
    with open(img1_path, "rb") as f:
        img1 = f.read()
    # os.remove(img1_path)
    with open(img2_path, "rb") as f:
        img2 = f.read()
    # os.remove(img2_path)
    with open(img3_path, "rb") as f:
        img3 = f.read()
    
    return _json_response_maker("success", {
        "img1": base64.b64encode(img1).decode(),
        "img2": base64.b64encode(img2).decode(),
        "img3": base64.b64encode(img3).decode()
    })
    

_work_stopped = False  
def _image_processing():
    """
    The image processing thread.
    """
    import cv2
    from NonDeepMethodBaseline import pyheal
    global _work_stopped
    while not _work_stopped:
        if _process_queue.empty():
            time.sleep(1)
            continue
        
        try:
            id = _process_queue.get()
            _process_queue.task_done()
            img = cv2.imread(os.path.join(_img_buffer_path, id, "img.png"))
            mask = cv2.imread(os.path.join(_img_buffer_path, id, "mask.png"))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img1_path = os.path.join(_img_buffer_path, id, "inpainted1.png")
            img2_path = os.path.join(_img_buffer_path, id, "inpainted2.png")
            img3_path = os.path.join(_img_buffer_path, id, "inpainted3.png")

            pyheal_img = img.copy()
            # Requires inplace assignment
            pyheal.inpaint(pyheal_img, mask.astype(bool, copy=True), 5)
            cv2.imwrite(img1_path, pyheal_img)
            cv2.imwrite(img2_path, cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA))
            _inference_deep_method(os.path.join(_img_buffer_path, id, "img.png"), os.path.join(_img_buffer_path, id, "mask.png"), img3_path)

            assert os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(img3_path)
        except Exception as e:
            print(e)
            _img_error__list.append(id)
        finally:
            _img_buffer_list.remove(id)


def _inference_deep_method(image_path, mask_path, out_path):
    data_analysis2.test(None, image_path, mask_path, out_path)



if __name__ == '__main__':
    from threading import Thread
    import atexit
    process = Thread(target=_image_processing)
    process.start()

    def shutdown():
        global _work_stopped
        _work_stopped = True
        process.join()
    atexit.register(shutdown)

    app.run(host='localhost', port=5000, debug=True)