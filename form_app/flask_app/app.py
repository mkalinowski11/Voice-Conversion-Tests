from flask import Flask, request, render_template, flash, redirect, url_for
import os
from flask_sqlalchemy import SQLAlchemy

VOICE_DATA = os.path.join('static', 'voice_data')
app = Flask(__name__)
app.secret_key = 'abracadabra'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///form.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Response(db.Model):
    __tablename__ = 'responses'
    _id = db.Column(db.Integer, primary_key = True, autoincrement=True)
    resp1 = db.Column(db.Integer)
    resp2 = db.Column(db.Integer)
    resp3 = db.Column(db.Integer)

    def __init__(self, resp1, resp2, resp3):
        self.resp1 = resp1
        self.resp2 = resp2
        self.resp3 = resp3

    def __repr__(self):
        return f'{self.resp1}, {self.resp2}, {self.resp3}'

def parse_answers(answers):
    if not any(map(lambda a : len(a) == 0, answers)):
        return True
    else:
        return False

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = []
        for spk_id in os.listdir(VOICE_DATA):
            data.append(request.form.get(spk_id))
        if parse_answers(data):
            data = list(map(lambda a: int(a), data))
            response = Response(*data)
            db.session.add(response)
            db.session.commit()
            print(f'Succesfully added data: {data}')
    return render_template(
            "voice_form.html",
            title="Weryfikacja jakości konwersji głosu",
            speaker_data = os.listdir(VOICE_DATA),
            samples = os.listdir(os.path.join(
                VOICE_DATA, os.listdir(VOICE_DATA)[0]
            ))
        )

if __name__ == "__main__":
    if not os.path.isfile(app.config['SQLALCHEMY_DATABASE_URI']):
        db.create_all()
    app.run(debug=True)