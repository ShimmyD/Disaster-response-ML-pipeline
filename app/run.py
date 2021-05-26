import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    stop_words = stopwords.words('english')
    text=text.lower()
    text = re.sub(r'[^\w\s]','',text)
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[lemmatizer.lemmatize(token).strip() for token in tokens if token not in stop_words]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names=df.columns[4:]
    category_counts=df[df.columns[4:]].sum().values
    dic=dict(zip(category_names,category_counts))
    dic_sort=sorted(dic.items(),key=lambda kv:kv[1],reverse=True)
    category_names=[i[0] for i in dic_sort]
    category_counts=[i[1] for i in dic_sort]
    
    vect=CountVectorizer(tokenizer=tokenize)
    vector=vect.fit_transform(df['message'].values)
    count=vector.toarray().sum(axis=0)
    words_lst=vect.get_feature_names()

    words_dic=dict(zip(words_lst,count))

    top_10=sorted(words_dic.items(), key=lambda kv: kv[1],reverse=True)[:10]
    words=[i[0] for i in top_10]
    counts=[i[1] for i in top_10]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of disaster categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=counts
                )
            ],

            'layout': {
                'title': 'Distribution of top 10 words in message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()