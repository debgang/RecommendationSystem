from email import header
from operator import index
from flask import Flask, request, render_template, jsonify
from model import getsentimentRecommendation


app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    # get user from the html form
    user = request.form['userName']
    # convert text to lowercase
    user = user.lower()
    items = getsentimentRecommendation(user)

    if(not(items is None)):
        print(f"retrieving items....{len(items)}")
        print(items)
        # data=[items.to_html(classes="table-striped table-hover", header="true",index=False)
        return render_template(
            "index.html",
            column_names=list(items.columns.values),  # Convert to list
            row_data=list(items.values.tolist()),
            zip=zip
        )
    else:
        return render_template("index.html", message="User Name doesn't exists, No product recommendations at this point of time!")


if __name__ == '__main__':
    app.run()