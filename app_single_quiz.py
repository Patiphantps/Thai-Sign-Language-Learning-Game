from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import random
import time
import numpy as np
import os
import urllib.parse
from flask import session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from PIL import ImageFont, ImageDraw, Image
from flask import Flask, render_template, request, redirect, url_for, session
from utils import load_models, preprocess_image
from flask import send_from_directory
from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from flask import flash, get_flashed_messages
from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from math import ceil


app = Flask(__name__)
app.secret_key = 'my_sign_language_game_2025'
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
models = load_models()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scores.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

labels = {
    "life": ["ไป", "ง่วงนอน", "ทำงาน", "กิน", "พัก", "เรียน", "เที่ยว", "เดิน", "ออกกำลังกาย", "ทำอาหาร"],
    "greeting": ["สวัสดี", "สบายดี", "ขอบคุณ"],
    "family": ["พ่อ", "แม่", "ลูก", "พี่"],
    "feeling": ["รัก", "กังวล", "เสียใจ", "กลัว"]
}

correct_answered = False



class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    
class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    player_name = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    
with app.app_context():
    db.create_all()



@app.route('/save_and_home')
def save_and_home():
    session['saved_score'] = quiz_state['score']
    session['saved_category'] = quiz_state['category']
    session['saved_count'] = quiz_state['count']
  
    
    return redirect(url_for('home'))

@app.route('/leaderboard')
def leaderboard():
    page = int(request.args.get('page', 1))
    per_page = 10
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

   
    players = db.session.query(Score.player_name).filter(Score.player_name != 'guest').distinct().all()
    leaderboard_data = []

    for (player_name,) in players:
        total = db.session.query(db.func.sum(Score.score)) \
            .filter(Score.player_name == player_name).scalar() or 0

        category_scores = {}
        categories = ['life', 'greeting', 'family', 'feeling']
        for cat in categories:
            cat_score = db.session.query(db.func.sum(Score.score)) \
                .filter(Score.player_name == player_name, Score.category == cat).scalar() or 0
            category_scores[cat] = cat_score

        leaderboard_data.append((player_name, total, category_scores))

    
    leaderboard_data.sort(key=lambda x: x[1], reverse=True)

    total_players = len(leaderboard_data)
    total_pages = ceil(total_players / per_page)

   
    paginated_data = leaderboard_data[start_idx:end_idx]

    return render_template(
        'leaderboard.html',
        leaderboard=paginated_data,
        current_page=page,
        total_pages=total_pages
    )




@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route('/action/<path:filename>')
def serve_file(filename):
    return send_from_directory('action', filename)

@app.route('/action/video/<path:filename>')
def serve_video(filename):
    decoded_filename = urllib.parse.unquote(filename)  
    return send_from_directory('action', decoded_filename)


@app.route('/category')
def category():
    selected_category = request.args.get('category')
    page = int(request.args.get('page', 1))  
    per_page = 4  

    words = labels.get(selected_category, []) if selected_category else []

    total_pages = (len(words) + per_page - 1) // per_page
    paginated_words = words[(page - 1) * per_page : page * per_page]
    
    top5_players = (
    db.session.query(Score.player_name, db.func.sum(Score.score).label('total_score'))
    .group_by(Score.player_name)
    .order_by(db.desc('total_score'))
    .limit(5)
    .all()
)

    return render_template(
    'category.html',
    labels=labels,
    selected_category=selected_category,
    words=paginated_words,
    page=page,
    total_pages=total_pages,
    top5_players=top5_players 
)



quiz_state = {
    "category": None,
    "word": None,
    "count": 0,
    "score": 0,
    "done": False,
    "remaining_words": []
}


def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL)''')
    conn.commit()
    conn.close()
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            flash('เข้าสู่ระบบสำเร็จ', 'success')
            return redirect(url_for('home'))
        else:
            flash('ชื่อผู้ใช้หรือรหัสผ่านผิด', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            flash('สมัครสำเร็จ! ', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('ชื่อผู้ใช้นี้มีอยู่แล้ว', 'danger')
        conn.close()
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    
    return redirect(url_for('home'))





@app.route('/start', methods=['POST'])
def start():
    
    if session.get('user_id'):
        session['player_name'] = session['username']
        return redirect(url_for('category'))
    
    
    name = request.form.get('name')
    if not name:
        flash("กรุณาใส่ชื่อก่อนเริ่มเล่นแบบ Guest")
        return redirect(url_for('home'))

    session['player_name'] = name
    return redirect(url_for('category'))


@app.route('/set_name', methods=['POST'])
def set_name():
    name = request.form['name'].strip()
    if not name:
        flash("❗ กรุณากรอกชื่อก่อนเริ่มเล่น")
        return redirect(url_for('home'))

    
    session['player_name'] = name
    session['is_guest'] = True  

    return redirect(url_for('category'))





@app.route('/start_quiz', methods=["POST"])
def start_quiz():
    reset_quiz_state()  
    
    category = request.form.get("category")
    page = int(request.form.get("page", 1))
    per_page = 4

    if category not in labels:
        return "Invalid category", 400

    all_words = labels[category]
    paginated_words = all_words[(page - 1) * per_page : page * per_page]

    if not paginated_words:
        return "No words for quiz", 400

    quiz_state["category"] = category
    quiz_state["total_words"] = len(paginated_words)
    quiz_state["words"] = paginated_words

    return redirect(url_for("next_quiz"))


def reset_quiz_state():
    global quiz_state
    quiz_state = {
        "category": None,
        "word": None,
        "count": 0,
        "score": 0,
        "done": False,
        "remaining_words": [],
        "total_words": 0,
        "used_words": set(),
        "time_left": 120  
    }





@app.route('/next_quiz')
def next_quiz():
    global correct_answered
    if quiz_state["count"] >= quiz_state["total_words"]:
        quiz_state["done"] = True
        return redirect(url_for("summary"))

    available_words = [w for w in quiz_state["words"] if w not in quiz_state["used_words"]]
    if not available_words:
        quiz_state["done"] = True
        return redirect(url_for("summary"))

    word = random.choice(available_words)
    quiz_state["word"] = word
    quiz_state["used_words"].add(word)  
    quiz_state["counted"] = False
    correct_answered = False

    return render_template(
        "quiz_single.html",
        word=word,
        score=quiz_state["score"],
        count=quiz_state["count"],
        total=quiz_state["total_words"],
        correct_answered=correct_answered
        
    )






@app.route('/summary')
def summary():
    if quiz_state["score"] > 0 and session.get('user_id'):
        existing_score = Score.query.filter_by(
            player_name=session.get("username"),
            category=quiz_state["category"]
        ).first()

        if existing_score:
            
            existing_score.score = quiz_state["score"]
        else:
           
            new_score = Score(
                player_name=session.get("username"),
                category=quiz_state["category"],
                score=quiz_state["score"]
            )
            db.session.add(new_score)

        db.session.commit()
        print(" บันทึกคะแนนลง database แล้ว!")

    return render_template("summary.html", score=quiz_state["score"], total=quiz_state["total_words"])






@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def draw_thai_text(img, text, position, font_path='font/THSarabunNew.ttf', font_size=32, color=(0, 255, 0), box=False, box_color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(font_path, font_size)

    if box:
        bbox = draw.textbbox(position, text, font=font)
        padding = 10
        box_coords = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
        draw.rectangle(box_coords, outline=box_color, width=4)

    draw.text(position, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def gen_frames():
    global correct_answered

    category = quiz_state["category"]
    word = quiz_state["word"]
    model = models[category]
    label_list = labels[category]

    quiz_state["counted"] = False
    correct_answered = False
    correct_streak = 0
    STREAK_THRESHOLD = 2
    CONFIDENCE_THRESHOLD = 70

    previous_crop = None
    last_predict_time = time.time()
    predict_wait_sec = 2.5
    difference_threshold = 10

    correct_display_start = None  

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        imgHeight, imgWidth, _ = frame.shape
        frame_width = 350
        frame_height = 350  

        x1 = int((imgWidth - frame_width) / 2)
        y1 = int((imgHeight - frame_height) / 2)
        x2 = x1 + frame_width
        y2 = y1 + frame_height

       
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = draw_thai_text(frame, "ยกมือให้อยู่ในกรอบ", (x1 + 10, y1 - 30), font_size=28, color=(0, 255, 255))

       
        imgCrop = frame[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)

        predicted_word = ""
        accuracy = 0
        result_text = "❌ ลองใหม่"

        if previous_crop is None:
            previous_crop = gray_crop
            last_predict_time = time.time()
        else:
            diff_score = np.mean(cv2.absdiff(previous_crop, gray_crop))
            previous_crop = gray_crop

            if diff_score > difference_threshold:
                last_predict_time = time.time()

            if time.time() - last_predict_time >= predict_wait_sec:
                img_resized = cv2.resize(imgCrop, (224, 224))
                img_normalized = img_resized.astype("float32") / 255.0
                img_expanded = np.expand_dims(img_normalized, axis=0)

                prediction = model.predict(img_expanded)[0]
                predicted_index = np.argmax(prediction)
                predicted_word = label_list[predicted_index]
                accuracy = round(prediction[predicted_index] * 100, 2)

                if predicted_word == word and accuracy >= CONFIDENCE_THRESHOLD:
                    correct_streak += 1
                    result_text = "✅ ถูกต้อง"

                    if correct_streak >= STREAK_THRESHOLD and not quiz_state["counted"]:
                        quiz_state["score"] += 1
                        quiz_state["count"] += 1
                        quiz_state["counted"] = True
                        correct_answered = True
                        correct_display_start = time.time()

        
        if correct_answered and correct_display_start:
            if time.time() - correct_display_start <= 2:
                overlay = frame.copy()

                
                overlay = frame.copy()

               
                center_box_width = int(imgWidth *  1)
                center_box_height = int(imgHeight * 0.3)

                center_box_x1 = int((imgWidth - center_box_width) / 2)
                center_box_y1 = int((imgHeight - center_box_height) / 2)
                center_box_x2 = center_box_x1 + center_box_width
                center_box_y2 = center_box_y1 + center_box_height

               
                cv2.rectangle(overlay, (center_box_x1, center_box_y1), (center_box_x2, center_box_y2), (0, 255, 0), -1)

               
                alpha = 0.3
                frame[center_box_y1:center_box_y2, center_box_x1:center_box_x2] = cv2.addWeighted(
                    overlay[center_box_y1:center_box_y2, center_box_x1:center_box_x2],
                    alpha,
                    frame[center_box_y1:center_box_y2, center_box_x1:center_box_x2],
                    1 - alpha,
                    0
                )



                text = " ถูกต้อง!"
                font_path = 'font/THSarabunNew.ttf'
                font_size = 50

                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.truetype(font_path, font_size)
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                center_x = int((imgWidth - text_width) / 2)
                center_y = int((imgHeight - text_height) / 2)

                frame = draw_thai_text(frame, text, (center_x, center_y), font_size=font_size, color=(255, 255, 255))

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                continue

            else:
                break

      
        if predicted_word:
            frame = draw_thai_text(frame, f'{predicted_word} ({accuracy}%)', (30, 40))
        frame = draw_thai_text(frame, result_text, (30, 80), color=(255, 0, 0))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')








if __name__ == '__main__':
    init_db()
    app.run(debug=True)

