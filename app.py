from flask import Flask, render_template, request, Response, redirect, url_for
from utils import load_models, preprocess_image
from flask import send_from_directory
import cv2
import numpy as np
import time
import random 

import urllib.parse

app = Flask(__name__)
camera = cv2.VideoCapture(0)

models = load_models()
labels = {
    "life": ["ไป", "ง่วงนอน", "ทํางาน", "กิน"],
    "greeting": ["สวัสดี", "สบายดี", "ขอบคุณ"],
    "family": ["พ่อ", "แม่", "ลูก", "พี่"],
    "feeling": ["รัก", "กังวล", "เสียใจ", "กลัว"]
}

quiz_status = {
    "category": None,
    "score": 0,
    "count": 0,
    "done": False
}


@app.route('/')
def index():
    category = request.args.get('category', 'life')
    selected_words = labels.get(category, [])
    return render_template("index.html", words=selected_words, selected_category=category)

@app.route('/start_quiz_random', methods=["POST"])
def start_quiz_random():
    category = request.form.get("category")
    print("===> category ที่รับมา:", category)
    print("===> หมวดที่มีอยู่ใน labels:", list(labels.keys()))
    if not category or category not in labels:
        return "Invalid category", 400

    word_list = labels[category]
    selected_words = random.sample(word_list, min(5, len(word_list)))

    quiz_status["score"] = 0
    quiz_status["count"] = 0
    quiz_status["done"] = False
    quiz_status["word_list"] = selected_words
    quiz_status["category"] = category

    return redirect(url_for("quiz_page", index=0, category=category))

@app.route('/quiz/<int:index>/<category>')
def quiz_page(index, category):
    if index >= len(quiz_status["word_list"]):
        return redirect(url_for("summary"))

    word = quiz_status["word_list"][index]
    return render_template("quiz.html", expected_word=word, index=index, category=category)

@app.route('/next_quiz/<int:index>/<category>')
def next_quiz(index, category):
    return redirect(url_for("quiz_page", index=index + 1, category=category))




@app.route('/category')
def category():
    selected = request.args.get('category')
    if selected not in labels:
        return redirect('/')
    quiz_status["category"] = selected
    quiz_status["score"] = 0
    quiz_status["count"] = 0
    quiz_status["done"] = False
    words = labels[selected]
    return render_template("index.html", words=words, category=selected)



@app.route('/start_quiz/<category>/<word>')
def start_quiz(category, word):
    quiz_status["category"] = category
    return render_template("quiz.html", expected_word=word, category=category)


@app.route('/check_done')
def check_done():
    return {
        "done": quiz_status["done"],
        "score": quiz_status["score"],
        "count": quiz_status["count"]
    }


@app.route('/summary')
def summary():
    score = quiz_status["score"]
    quiz_status["score"] = 0
    quiz_status["count"] = 0
    quiz_status["done"] = False
    return render_template("summary.html", score=score)


@app.route('/video/<word>')
def video_feed(word):
    return Response(gen_frames(word), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/action/video/<path:filename>')
def serve_video(filename):
    decoded_filename = urllib.parse.unquote(filename)  # ถอดรหัส URL เช่น %E0%B9%84%E0%B8%9B -> ไป
    return send_from_directory('action', decoded_filename)


def gen_frames(word):
    category = quiz_status["category"]
    model = models[category]
    label_list = labels[category]
    counted = False
    last_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            break

        input_img = preprocess_image(frame)
        prediction = model.predict(input_img)[0]
        predicted_index = np.argmax(prediction)
        predicted_word = label_list[predicted_index]
        accuracy = round(prediction[predicted_index] * 100, 2)

        result_text = "❌ ลองใหม่"

        if predicted_word == word:
            result_text = "✅ ถูกต้อง"
            if not counted:
                quiz_status["score"] += 1
                quiz_status["count"] += 1
                counted = True
                last_time = time.time()

            if quiz_status["count"] >= 5:
                quiz_status["done"] = True
                break

        if counted and time.time() - last_time > 3:
            counted = False

        cv2.putText(frame, f'{predicted_word} ({accuracy}%)', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, result_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    app.run(debug=True)
