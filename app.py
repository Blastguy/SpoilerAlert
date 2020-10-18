from flask import Flask, render_template, request
from model import Test

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    spoilertext = request.form.get('spoilertext')
    # Run ML on spoilertext
    # print(spoilertext)

    spoiler = False
    spoilersArray = ["kill", "spoiler", "die"]
    for s in spoilersArray:
        if s in spoilertext:
            spoiler = True
        
    data = {
        "safe": "Safe",
    }

    if spoiler:
        data = {
            "spoiler": "Spoiler",
        }

    return render_template('submit.html', data=data)


if __name__ == '__main__':
    app.run()
