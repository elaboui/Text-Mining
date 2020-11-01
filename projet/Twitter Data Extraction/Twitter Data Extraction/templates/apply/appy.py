

from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
import twitter_credentials as t

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
    print(t.ACCESS_TOKEN)
 

    return render_template('index.html',a=t.ACCESS_TOKEN)
if __name__ == '__main__':
    app.run()
