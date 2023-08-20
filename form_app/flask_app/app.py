from flask import Flask, request, render_template, flash, redirect, url_for
import os
# from flask_sqlalchemy import SQLAlchemy

VOICE_DATA = os.path.join('static', 'voice_data')
app = Flask(__name__)
app.secret_key = 'abracadabra' 

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
            print(f'Succesfully added data: {data}')
        else:
            print("Not ok")
    return render_template(
            "form.html",
            title="Weryfikacja jakości głosu",
            speaker_data = os.listdir(VOICE_DATA),
            samples = os.listdir(os.path.join(
                VOICE_DATA, os.listdir(VOICE_DATA)[0]
            )),
            labels = [
                'Źródło A',
                'Głos docelowy B',
                'Konwersja A->B',
                'Konwersja B->A'
            ]
        )

if __name__ == "__main__":
    app.run(debug=True)