import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import plotly
import plotly.graph_objs as go
import nltk
import json
import faculty
import drawFigure
from flask import Flask, render_template, redirect, request, session
from flask_sqlalchemy import SQLAlchemy
import bcrypt

# Download NLTK data
nltk.download('vader_lexicon')
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


with app.app_context():
    db.create_all()
# Initialize VADER
sid = SentimentIntensityAnalyzer()


# Function to perform sentiment analysis with VADER
def perform_sentiment_analysis(student_feedback):
    sentiments = []
    for feedback in student_feedback:
        scores = sid.polarity_scores(str(feedback))
        sentiment_score = scores['compound']  # Use compound score as overall sentiment
        sentiments.append(sentiment_score)

    # Count positive, negative, and neutral sentiments
    positive_count = sum(1 for score in sentiments if score > 0)
    negative_count = sum(1 for score in sentiments if score < 0)
    neutral_count = len(sentiments) - positive_count - negative_count

    # Return the sentiment counts
    return positive_count, negative_count, neutral_count


with open('feedback1.json') as file:
    json_string = file.read()
    documents1 = json.loads(json_string)

with open('feedback2.json') as file:
    json_string = file.read()
    documents2 = json.loads(json_string)

with open('feedback3.json') as file:
    json_string = file.read()
    documents3 = json.loads(json_string)

with open('feedback4.json') as file:
    json_string = file.read()
    documents4 = json.loads(json_string)

with open('feedback5.json') as file:
    json_string = file.read()
    documents5 = json.loads(json_string)

with open('feedback6.json') as file:
    json_string = file.read()
    documents6 = json.loads(json_string)

label2category = {1: 'positive', 0: 'neutral', -1: 'negative'}
category2label = {cat: label for label, cat in label2category.items()}

categories1 = [category2label[category] for doc, category in documents1]
categories2 = [category2label[category] for doc, category in documents2]
categories3 = [category2label[category] for doc, category in documents3]
categories4 = [category2label[category] for doc, category in documents4]
categories5 = [category2label[category] for doc, category in documents5]
categories6 = [category2label[category] for doc, category in documents6]

corpus1 = [' '.join(document) for document, cat in documents1]
corpus2 = [' '.join(document) for document, cat in documents2]
corpus3 = [' '.join(document) for document, cat in documents3]
corpus4 = [' '.join(document) for document, cat in documents4]
corpus5 = [' '.join(document) for document, cat in documents5]
corpus6 = [' '.join(document) for document, cat in documents6]


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/get')
def home():
    return render_template('home.html')


@app.route('/Register', methods=['GET', 'POST'])
def Register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/Login')

    return render_template('Register.html')


@app.route('/Login', methods=['GET', 'POST'])
def Login():
    if request.method == 'POST':
        email = request.form['email']

        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/Dashboard')
        else:
            return render_template('Login.html', error='Invalid user')

    return render_template('Login.html')


@app.route('/Dashboard')
def Dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('Dashboard.html', user=user)
    return redirect('/Login')


@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect('/Login')


@app.route('/Let start this')
def Let():
    return render_template('index.html')


@app.route('/')
def display():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def caption():
    if request.method == 'POST':

        f = request.files["file_name"]
        path = "./static/{}".format(f.filename)
        f.save(path)

        category_no = int(request.form['Cate'])

        df = pd.read_csv(path)
        cols1 = []
        cols2 = []
        cols3 = []
        cols4 = []
        cols5 = []
        cols6 = []

        substring1 = ['teacher', 'faculty', 'feedback', 'effectiveness', 'teaching', 'knowledge', 'delivery', 'content',
                      'quality',
                      'lecture', 'subject', 'syllabus', 'review', 'assessment']
        substring2 = ['course', 'content', 'syllabus', 'review', 'evaluation', 'curriculum', 'syllabi', 'contents',
                      'level',
                      'difficulty', 'lecture', 'outline']
        substring3 = ['exam', 'examination', 'pattern', 'conduct', 'question', 'paper', 'level', 'outline']
        substring4 = ['laboratory', 'laboratories', 'lab', 'facility', 'facilities', 'review', 'feedback', 'rate',
                      'learning']
        substring5 = ['library', 'facilities', 'books', 'availability', 'facility', 'material', 'rate', 'feedback',
                      'review']
        substring6 = ['extra', 'curricular', 'activity', 'activities']

        for i in list(df.columns):
            for j in substring1:
                if j.casefold() in i.casefold():
                    cols1.append(df.columns.get_loc(i))
                    if cols1 != []:
                        break

        for i in list(df.columns):
            for j in substring2:
                if j.casefold() in i.casefold():
                    cols2.append(df.columns.get_loc(i))
                    if cols2 != []:
                        break

        for i in list(df.columns):
            for j in substring3:
                if j.casefold() in i.casefold():
                    cols3.append(df.columns.get_loc(i))
                    if cols3 != []:
                        break

        for i in list(df.columns):
            for j in substring4:
                if j.casefold() in i.casefold():
                    cols4.append(df.columns.get_loc(i))
                    if cols4 != []:
                        break

        for i in list(df.columns):
            for j in substring5:
                if j.casefold() in i.casefold():
                    cols5.append(df.columns.get_loc(i))
                    if cols5 != []:
                        break

        for i in list(df.columns):
            for j in substring6:
                if j.casefold() in i.casefold():
                    cols6.append(df.columns.get_loc(i))
                    if cols6 != []:
                        break

        cols = cols1 + cols2 + cols3 + cols4 + cols5 + cols6
        cols = list(set(cols))
        df_form = pd.read_csv(path, usecols=cols)
        reviews = np.array(df_form)

        # Check if the reviews array is empty
        if reviews.size == 0:
            print("The 'reviews' array is empty. Please check your data.")
            # Add appropriate handling or return from the function.
        else:
            # Continue with the rest of your code for sentiment prediction
            # ...

            # Example: Print the first few rows of the DataFrame
            print(df_form.head())
        pos1, n1, neg1 = 0, 0, 0
        # Call the faculty.predict method
        try:
            pos1, n1, neg1 = faculty.predict(corpus1, categories1, reviews[:, 0])
        except Exception as e:
            print(f"Error predicting for column 1: {e}")

        # Initialize variables with default values
        pos2, n2, neg2 = 0, 0, 0

        # Call the faculty.predict method
        try:
            pos2, n2, neg2 = faculty.predict(corpus1, categories1, reviews[:, 1])
        except Exception as e:
            print(f"Error predicting for column 2: {e}")

        # Initialize variables with default values for the third column
        pos3, n3, neg3 = 0, 0, 0

        # Call the faculty.predict method for the third column
        try:
            pos3, n3, neg3 = faculty.predict(corpus1, categories1, reviews[:, 2])
        except Exception as e:
            print(f"Error predicting for column 3: {e}")

        # Initialize variables with default values for the third column
        pos4, n4, neg4 = 0, 0, 0

        # Call the faculty.predict method for the third column
        try:
            pos4, n4, neg4 = faculty.predict(corpus1, categories1, reviews[:, 3])
        except Exception as e:
            print(f"Error predicting for column 4: {e}")

        # Initialize variables with default values for the third column
        pos5, n5, neg5 = 0, 0, 0

        # Call the faculty.predict method for the third column
        try:
            pos5, n5, neg5 = faculty.predict(corpus1, categories1, reviews[:, 4])
        except Exception as e:
            print(f"Error predicting for column 5: {e}")

        # Initialize variables with default values for the third column
        pos6, n6, neg6 = 0, 0, 0

        # Call the faculty.predict method for the third column
        try:
            pos6, n6, neg6 = faculty.predict(corpus1, categories1, reviews[:, 5])
        except Exception as e:
            print(f"Error predicting for column 6: {e}")

        results = {
            'f1': 'Teacher Feedback',
            'pos1': pos1,
            'n1': n1,
            'neg1': neg1,
            'f2': 'Course Content',
            'pos2': pos2,
            'n2': n2,
            'neg2': neg2,
            'f3': 'Examination pattern',
            'pos3': pos3,
            'n3': n3,
            'neg3': neg3,
            'f4': 'Laboratory',
            'pos4': pos4,
            'n4': n4,
            'neg4': neg4,
            'f5': 'Library Facilities',
            'pos5': pos5,
            'n5': n5,
            'neg5': neg5,
            'f6': 'Extra Co-Curricular Activities',
            'pos6': pos6,
            'n6': n6,
            'neg6': neg6,
        }
        values = list([[pos1, n1, neg1], [pos2, n2, neg2], [pos3, n3, neg3], [pos4, n4, neg4], [pos5, n5, neg5],
                       [pos6, n6, neg6]])
        labels = list(['Teacher Feedback', 'Course Content', 'Examination pattern', 'Laboratory', 'Library Facilities',
                       'Extra Co-Curricular Activities'])

        print(values[category_no - 1], labels[category_no - 1], category_no, category_no - 1)

        if category_no == 1:
            results_1 = {
                'f1': 'Teacher Feedback',
                'pos1': pos1,
                'n1': n1,
                'neg1': neg1
            }

            drawFigure.make(values[category_no - 1], labels[category_no - 1], category_no)

            return render_template('index1.html', result1=results_1, cat=category_no)


        elif category_no == 2:
            results_2 = {
                'f2': 'Course Content',
                'pos2': pos2,
                'n2': n2,
                'neg2': neg2
            }

            drawFigure.make(values[category_no - 1], labels[category_no - 1], category_no)

            return render_template('index1.html', result2=results_2, cat=category_no)


        elif category_no == 3:
            results_3 = {
                'f3': 'Examination pattern',
                'pos3': pos3,
                'n3': n3,
                'neg3': neg3
            }

            drawFigure.make(values[category_no - 1], labels[category_no - 1], category_no)

            return render_template('index1.html', result3=results_3, cat=category_no)


        elif category_no == 4:
            results_4 = {
                'f4': 'Laboratory',
                'pos4': pos4,
                'n4': n4,
                'neg4': neg4
            }

            drawFigure.make(values[category_no - 1], labels[category_no - 1], category_no)

            return render_template('index1.html', result4=results_4, cat=category_no)


        elif category_no == 5:
            results_5 = {
                'f5': 'Library Facilities',
                'pos5': pos5,
                'n5': n5,
                'neg5': neg5
            }

            drawFigure.make(values[category_no - 1], labels[category_no - 1], category_no)

            return render_template('index1.html', result5=results_5, cat=category_no)


        elif category_no == 6:
            results_6 = {
                'f6': 'Extra Co-Curricular Activities',
                'pos6': pos6,
                'n6': n6,
                'neg6': neg6
            }

            drawFigure.make(values[category_no - 1], labels[category_no - 1], category_no)

            return render_template('index1.html', result6=results_6, cat=category_no)


        else:
            for i in range(0, 6):
                fig = plt.figure(figsize=(8, 8), edgecolor='red', linewidth=10)
                plt.bar(x=['Positive', 'Neutral', 'Negative'], height=values[i], color=['blue', 'gold', 'red'])
                plt.title(labels[i], fontsize=24, weight='demibold', pad=15, fontstyle='italic', family='cursive')
                plt.xticks(rotation=0, fontsize=16)
                plt.yticks([])
                plt.xlabel('Feedback Type', fontsize=18, labelpad=17, weight=550, family='cursive')
                plt.ylabel('')
                fig.subplots_adjust(bottom=0.14)
                ax = plt.gca()
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                for p in ax.patches:
                    ax.annotate("%.1f%%" % (100 * float(p.get_height() / sum(values[i]))),
                                (p.get_x() + p.get_width() / 2., abs(p.get_height())),
                                ha='center', va='bottom', color='black', xytext=(0, 5), rotation='horizontal',
                                textcoords='offset points', fontsize=16, fontweight='medium')
                plt.savefig(f'./static/plot{i + 10}.jpg')

            return render_template('index1.html', result=results)

    else:
        return render_template('error.html')


@app.route('/More')
def index2():
    return render_template('index2.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        # Read CSV file
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')

            teacher_name = request.form['teacher']
            department = request.form['department']

            # Filter dataset based on selected teacher and department
            teacher_df = df[(df['Teacher'] == teacher_name) & (df['Department'] == department)]

            # Perform sentiment analysis on the filtered dataset using VADER
            positive_count, negative_count, neutral_count = perform_sentiment_analysis(teacher_df['student_feedback'])

            # Create a pie chart
            labels = ['Positive', 'Negative', 'Neutral']
            values = [positive_count, negative_count, neutral_count]
            colors = ['green', 'red', 'blue']

            graph = go.Pie(labels=labels, values=values, marker=dict(colors=colors))

            graphJSON = json.dumps([graph], cls=plotly.utils.PlotlyJSONEncoder)

            # Render template with results
            return render_template('results.html', graphJSON=graphJSON, positive_count=positive_count,
                                   negative_count=negative_count, neutral_count=neutral_count)

        else:
            return render_template('index2.html', error="Please upload a CSV file")


@app.route('/next')
def thankyou():
    return render_template('thankyou.html')


@app.route('/logout')
def Logout():
    return render_template('Login.html')


@app.route('/contact me')
def contact_us():
    return render_template('contact_us.html')


if __name__ == '__main__':
    app.run(debug=True)



