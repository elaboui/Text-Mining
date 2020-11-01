#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:45:09 2019

@author: ZOUITNI
"""
import tweepy
import tweepy as tw
from tweepy.streaming import StreamListener
from flask import Flask,redirect,render_template,request,url_for,jsonify
from tweepy import Stream
from info import data
import helpers
from analyzer import Analyzer
import threading,csv,re
import matplotlib.pyplot as plt
from datetime import datetime
from textblob import TextBlob
from config import consumer_key,consumer_secret,access_token,access_token_secret
application = Flask(__name__)

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

class Thread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        loc()

prev_list = list()

@application.route('/geo')
def geo():
    t = Thread()
    t.start()
    return render_template('geo.html')

@application.route('/data')
def stream_data():
    '''
    GeoJSON Format
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [125.6, 10.1]
      },
      "properties": {
        "name": "Dinagat Islands"
      }
    }
    '''
    new_data = []
    for item in data:
        temp_data = {}
        temp_data['geometry'] = {}
        temp_data['properties'] = {}
        temp_data['type'] = 'Feature'
        temp_data['geometry']['type'] = 'Point'
        temp_data['geometry']['coordinates'] = item['coordinates']['coordinates']
        temp_data['properties']['name'] = item['text']
        new_data.append(temp_data)
    global prev_list
    if prev_list == []:
        prev_list = new_data[:]
        return jsonify(new_data)
    else:
        d = []
        for item in new_data:
            if item not in prev_list:
                d.append(item)
        prev_list = new_data[:]
        return jsonify(d)
    
class listener(StreamListener):
    def on_connect(self):
        print('Stream starting...')

    def on_status(self, status):
        if status.geo is not None:
            t = dict()
            t['text'] = status.text
            t['coordinates'] = status.coordinates
            data.append(t)

    def on_error(self, status):
        print(status)

def loc():

    twitterStream = Stream(auth, listener())
    twitterStream.filter(locations=[
        -130.78125, -31.3536369415, 140.625, 63.8600358954
    ])


@application.template_filter()
def number(value):
    return "{:.4f}".format(value)

@application.template_filter()
def time(value):
    value = value[-4:] + ' ' + value[:19]
    return value

@application.route("/")
def index():
    
    return render_template("index1.html")


@application.route('/suggestion')
def suggestion():
    submit = request.args.get("submit", "")
    submit1 = request.args.get("submit1", "")
    submit2 = request.args.get("submit2", "")
    post = request.args.get("post", "")
    downld=request.args.get("download", "")
    ploter=request.args.get("plot", "")
    prog=request.args.get("progress", "")

    if  submit:
        a=analyse()
        return a
        
    elif submit1:
        v=visualisation()
        return v 
    elif submit2:
        g=geo()
        return g 
    elif post:
        pst= poster()
        return pst
    elif downld:
        sa = SentimentAnalysis()
        download= sa.DownloadData()
        return download
    elif ploter:
        keyword = request.args.get("screen_name", "")
        plo= getData(keyword)
        return plo
    elif prog:
        pro= progress()
        return pro
    
        
@application.route('/analyse')
def analyse():
    screen_name = request.args.get("screen_name", "")
    if not screen_name:
        return redirect(url_for("index"))

    tweets = helpers.get_user_timeline(screen_name, 200)

    positives = "static/SentiWS_v18c_Positive.txt"
    negatives = "static/SentiWS_v18c_Negative.txt"
    poENG = "static/positive-words.txt"
    neENG = "static/negative-words.txt"

    analyzer = Analyzer(positives, negatives, poENG, neENG)
    
    positive = 0
    negative = 0

    for tweet in tweets:

        tweet['score'] = analyzer.analyze(tweet['tweet'])

        if tweet['score'] > 0.0:
            positive += tweet['score']
        elif tweet['score'] < 0.0:
            negative -= tweet['score']

    chart = helpers.chart(positive, negative)
    return render_template("analyse.html", chart=chart, screen_name=screen_name, tweets=tweets)


@application.route("/visualisation")
def visualisation():

    screen_name = request.args.get("screen_name", "")
    if not screen_name:
        return redirect(url_for("index"))

    tweets = helpers.get_user_timeline(screen_name, 200)

    positives = "static/SentiWS_v18c_Positive.txt"
    negatives = "static/SentiWS_v18c_Negative.txt"
    poENG = "static/positive-words.txt"
    neENG = "static/negative-words.txt"

    analyzer = Analyzer(positives, negatives, poENG, neENG)

    positive = 0
    negative = 0

    for tweet in tweets:

        tweet['score'] = analyzer.analyze(tweet['tweet'])

        if tweet['score'] > 0.0:
            positive += tweet['score']
        elif tweet['score'] < 0.0:
            negative -= tweet['score']

    chart = helpers.chart(positive, negative)

    # render results
    return render_template("visualisation.html", chart=chart, screen_name=screen_name, tweets=tweets)



@application.route('/poster')
def poster():
    screen_name = request.args.get("screen_name", "")
    if screen_name:
        api.update_status(screen_name)
    return redirect(url_for("index"))

class SentimentAnalysis:

    def __init__(self):
        self.tweets = []
        self.tweetText = []
        
    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())
    
    @application.route('/download')
    def DownloadData(self):
       
        # input for term to be searched and how many tweets to search
        searchTerm = request.args.get("screen_name", "")
       
        #◙NoOfTerms = int(input("Enter how many tweets to search: "))

        # searching for tweets
        self.tweets = tw.Cursor(api.search, q=searchTerm, lang = "en").items(10)
         # iterating through tweets fetched
        for tweet in self.tweets:
            #Append to temp so that we can store in csv later. I use encode UTF-8
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
        # Open/create a file to append data to
        csvFile = open("csvFile/"+ searchTerm+'.csv', 'a')
        
        # Use csv writer
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(self.tweetText)
        
        csvFile.close()
       
        return redirect(url_for("index"))
    def get_tweet_sentiment(self, tweet):
		
        analysis = TextBlob(self.cleanTweet(tweet))
		
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count=10):
		
        tweets = []
        try:
                fetched_tweets = api.search(q=query, count=count)
    
                for tweet in fetched_tweets:
    				
                    parsed_tweet = {}
    
                    parsed_tweet['text'] = tweet.text+"\n"
                    parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
    
                    if tweet.retweet_count > 0:
                        if parsed_tweet not in tweets:
                            tweets.append(parsed_tweet)
                        else:
                            tweets.append(parsed_tweet)
    
                return tweets
    
        except tweepy.TweepError as e:
                print("Error : " + str(e))

@application.route('/progress')
def progress():

        		query = request.args.get("screen_name", "")
        		count = 20
        
        		obj = SentimentAnalysis()
     
        		tweets = obj.get_tweets(query=query, count=count)

        		ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']

        		ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']

        		positive = "{0:.2f}".format(100 * len(ptweets) / len(tweets))

        		negative = "{0:.2f}".format(100 * len(ntweets) / len(tweets))

        		neutral = "{0:.2f}".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets))
        		results = [positive, negative, neutral, query, count]
        
        		
        		return render_template('progress.html', data = results)
    
@application.route('/getData')
def getData(keyword):
    
    
    numberOfTweets = 20

    polarity_list = []
    numbers_list = []
    number = 1

    for tweet in tweepy.Cursor(api.search, keyword, lang="en").items(numberOfTweets):
        try:
            analysis = TextBlob(tweet.text)
            analysis = analysis.sentiment
            polarity = analysis.polarity
            polarity_list.append(polarity)
            numbers_list.append(number)
            number = number + 1

        except tweepy.TweepError as e:
            print(e.reason)

        except StopIteration:
            break


    axes = plt.gca()
    axes.set_ylim([-1, 2])

    plt.scatter(numbers_list, polarity_list)

    averagePolarity = (sum(polarity_list))/(len(polarity_list))
    averagePolarity = "{0:.0f}%".format(averagePolarity * 100)
    time  = datetime.now().strftime("At: %H:%M\nOn: %m-%d-%y")

    plt.text(0, 1.25, "Average Sentiment:  " + str(averagePolarity) + "\n" + time, fontsize=12, bbox = dict(facecolor='none', edgecolor='black', boxstyle='square, pad = 1'))

    plt.title("Sentiment of " + keyword + " on Twitter")
    plt.xlabel("Number of Tweets")
    plt.ylabel("Sentiment")
    plt.savefig('static/images/'+keyword+'.png')
    return render_template("plot.html",name = 'new_plot', url ='static/images/'+keyword+'.png',screen_name=keyword)


if __name__ == "__main__":
    
    application.run(threaded=True)
