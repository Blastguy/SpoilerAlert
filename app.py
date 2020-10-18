from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    spoilertext = request.form.get('spoilertext')
    # Run ML on spoilertext
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
