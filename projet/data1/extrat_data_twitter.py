# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:42:49 2020

@author: dell
"""

import os
import tweepy as tw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections
import nltk
from nltk.corpus import stopwords
import re
import networkx
import numpy as np
from nltk import bigrams
import networkx as nx
import warnings
from textblob import TextBlob
import time
from IPython.display import display


#First you will need define your keys

print("================== welcome ===============================")
print()
print()
print("entrer votre nom et prenom")
nom_prenom=input()
print("\n Hello "+nom_prenom+" est ce que vous aves deje un compte sur twitter devlopper? (y/n)")
rep=input()
if rep=="n":
    print()
    print("\n Veuiez visiter ce lien suivant:===>'https://developer.twitter.com/'")
    print()
    print()
    print("Thank you "+nom_prenom+" for your visiting")
else:
    print("****Remplir vos informations*********")
    print()
    #print("* consumer_key")
    consumer_key= '6aafmRHOMinQmOUxjzGG2AzoE'
    #input()#'6aafmRHOMinQmOUxjzGG2AzoE'
    print("* consumer_secret")
    consumer_secret='ExvFnoOb87IVwr4MjY7Jf9ZeecWbaTdeSs8RjYTTHquxUEEAa0'
    #input()# 'ExvFnoOb87IVwr4MjY7Jf9ZeecWbaTdeSs8RjYTTHquxUEEAa0'
    print("* access_token")
    access_token='1223301958754357250-6expxNC87AuWD5uSeuSNiyjDSycZnN'
    #input()# '1223301958754357250-6expxNC87AuWD5uSeuSNiyjDSycZnN'
    print("*access_token_secret")
    access_token_secret='GktEmBtjpmJToHdrefpafOQmly1fSVaONpu8gCWLV2dkX'
    #input()# 'GktEmBtjpmJToHdrefpafOQmly1fSVaONpu8gCWLV2dkX'
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    print("*************************************************************************************************************************")
    print()
    print("====== WELCOME TO DATA EXTRTCATION FOR TWITTER=======" )
    print()
    print("=== send tweet===")
    print("***creer #Hashtag ***")
    hashtag=input()
    api.update_status(hashtag)
    print("vous pouvez verfiez votre #hashtag est creer sur votre compte twitter\n")
    print("*************************************************************************************************************************")
    print()
    print("===> Search Twitter your Tweets ====")
    print()
    ############ t=int (200)
    ############ time.sleep(t)
    print("***Choose the language (for Arabic = 'ar', French = 'fr', English = 'en'):")
    lg=input()
    print("***Create the hashtag search:")
    search_words=input()
    print("***Create the date what to look for (exp:xxxxx-xx-xx):")
    date=input()
    print("***Create the number of tweets to search:")
    print()
    nbr=int(input())
    # Collect tweets
    tweets = tw.Cursor(api.search,
                       q=search_words,
                       lang=lg,
                       since=date).items(nbr)
    # print data
    for tweet in tweets:
        print(tweet.text)
    print("Collect a list of tweets")
    [tweet.text for tweet in tweets]
    #To Keep or Remove Retweets
    print()
    print("** voulez vous retweets your tweet?(y/n)")
    r=input()
    print("** voulez vous modifier le tweet?(y/n)")
    r1=input()
    if(r=="n") and (r1=="n"):
        print()
        print("==> Thank you for your answer")
    elif(r=="y") and (r1=="n"):
        #api.update_status(hashtag)
        api.destroy_status(hashtag.id)
        print("creer le txeet modifier")
        hashtag1=input()
        api.update_status(hashtag1)
    else:
        print("ecrire le retweets")
        new_search =input()
        api.update_status(new_search)
        print()
        print()
        # time.sleep(t)
        print("======= Who is Tweeting?========= ")
        # time.sleep(t)
        print()
        print("*** les personne a tweets *****")
        print("entrer le nombre a chercher")
        n=int(input())
        tweets = tw.Cursor(api.search, 
                           q=new_search,
                           lang=lg,
                           since=date).items(n)
        users_locs = [[tweet.user.screen_name, tweet.user.location] for tweet in tweets]
        users_locs
        print("*** Create a Pandas Dataframe From List of Tweet Data *** \n \n  ")
        tweet_text = pd.DataFrame(data=users_locs, 
                                  columns=['user', "location"])
        print(tweet_text)
        print("**************************************************************************************************************")
        # Analyze Word Frequency Counts Using Twitter Data and Tweepy in Python
        print()
        print("voulez vous continuer?(y/n)")
        rop=input()
        if rop =="n":
            print("\n Thank you for your visit")
        else:
            print()
            print("choisir une option par le Menu:")
            print("*******************************")
            print()
            print("       * 0. Exit")
            print("       * 1. Analyze Word Frequency")
            print("       * 2. Nettoyage de data")
            print("       * 3. Calculate and Plot Word Frequency")
            print("       * 4. Analyze Word Frequency")
            print("       * 5. Analyze Sentiments")
            print("       * 6. Visualization and basic statistics")
            m=int(input())
            print()
            print()
            #time.sleep(t)
            warnings.filterwarnings("ignore")
            sns.set(font_scale=1.5)
            sns.set_style("whitegrid")
            if(m==1):
                print("=========== Analyze Word Frequency ==================")
                print()
                print("*** Get Tweets Related *** ")
                print()
                print("* entrer la langue")
                lgg=input()
                print()
                print("* Entrer la date chercher")
                dt=input()
                print()
                print("* entrer le terme a chercher")
                search_term =input()
                print()
                print("* entrer le nombre des termes en relation a votre tweets")
                nbrr=int(input())
                print()
                tweets = tw.Cursor(api.search,
                                   q=search_term,
                                   lang=lgg,
                                   since=dt).items(nbrr)
                all_tweets = [tweet.text for tweet in tweets]
                all_tweets[:nbrr]
                print()
                print(all_tweets)
                print()
                print("**** this step has finished****")
                print()
                print("choisir une option par le Menu:")
                print("*******************************")
                print()
                print("       * 0. Exit")
                print("       * 1. Analyze Word Frequency")
                print("       * 2. Nettoyage de data")
                print("       * 3. Calculate and Plot Word Frequency")
                print("       * 4. Analyze Word Frequency")
                print("       * 5. Analyze Sentiments")
                print("       * 6. Visualization and basic statistics")
                m=int(input())
                print()
                print()
            if(m==2):
                print("=========== Nettoyage Data  ==================")
                print()
                print("*** Remove URLs (links) ***")
                print()
                def remove_url(txt):
                    """Replace URLs found in a text string with nothing 
                    (i.e. it will remove the URL from the string).
                    Parameters
                    ----------
                    txt : string
                    A text string that you want to parse and remove urls.
                    Returns
                    -------
                    The same txt string with url's removed.
                    """
                    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
                #After defining the function, you can call it in a list comprehension to create a list of the clean tweets.
                all_tweets_no_urls = [remove_url(tweet) for tweet in all_tweets]
                all_tweets_no_urls[:nbrr]
                
                print()
                #Text Cleanup - Address Case Issues
                print("*** Address Case Issues ***")
                print()
                set(all_tweets_no_urls)
                print(all_tweets_no_urls)
                # Make all elements in the list lowercase
                print("*** Make all elements in the list lowercase *** ")
                lower_case = [word.lower() for word in all_tweets_no_urls]
                # Get all elements in the list
                #print(lower_case)
                set(lower_case)
                print(lower_case)
                print("*** Split the words from one tweet into unique elements ***")
                print()
                # Split the words from one tweet into unique elements
                all_tweets_no_urls[0].split()
                print("*** Split the words from one tweet into unique elements ***")
                print()
                all_tweets_no_urls[0].lower().split()
                print(all_tweets_no_urls)
                print("*** Create a list of lists containing lowercase words for each tweet***")
                print()
                words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_urls]
                words_in_tweet[:nbrr]
                print(words_in_tweet)
                print("this step has finished")
                print()
                print("choisir une option par le Menu:")
                print("*******************************")
                print()
                print("       * 0. Exit")
                print("       * 1. Analyze Word Frequency")
                print("       * 2. Nettoyage de data")
                print("       * 3. Calculate and Plot Word Frequency")
                print("       * 4. Analyze Word Frequency")
                print("       * 5. Analyze Sentiments")
                print("       * 6. Visualization and basic statistics")
                m=int(input())
                print()
                print()
            if(m==3):
                print("=========== Calculate and Plot Word Frequency  ==================")
                print()
                print("*** List of all words across tweets ***")
                print()
                all_words_no_urls = list(itertools.chain(*words_in_tweet))
                print("*** Create counter ***")
                print()
                counts_no_urls = collections.Counter(all_words_no_urls)
                counts_no_urls.most_common(15)
                print(counts_no_urls)
                print()
                clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(15),
                                                    columns=['words', 'count'])
                print(clean_tweets_no_urls.head())
                print()
                print("--------------------------------------------------------")
                print("*** Menu modele of  graph *** ")
                print()
                print(" 0.Exit")
                print(" 1. Including All Words")
                print(" 2. Without Stop Words")
                print(" 3. Without Stop or Collection Words")
                m1=int(input())
                print()
                if m1==0:
                    print()
                    print("Menu principale")
                    print("-----------------")
                    print("       * 1. Analyze Word Frequency")
                    print("       * 2. Nettoyage de data")
                    print("       * 3. Calculate and Plot Word Frequency")
                    print("       * 4. Analyze Word Frequency")
                    print("       * 5. Analyze Sentiments")
                    m=int(input())
                    print()
                    print()
                if m1==1:
                    print("======= Including All Words =======")
                    print()
                    fig, ax = plt.subplots(figsize=(8, 8))
                    # Plot horizontal bar graph
                    clean_tweets_no_urls.sort_values(by='count').plot.barh(x='words',
                                                    y='count',
                                                    ax=ax,
                                                    color="purple")
                    ax.set_title("Common Words Found in Tweets (Including All Words)")
                    print()
                    plt.show()
                    print()
                    print("--------------------------------------------------------")
                    print()
                    print("*** Menu modele of  graph *** ")
                    print()
                    print(" 0.Exit")
                    print(" 1. Including All Words")
                    print(" 2. Without Stop Words")
                    print(" 3. Without Stop or Collection Words")
                    m1=int(input())
                    print()
                    #time.sleep(t)    
                if m1==2:
                     print("======= Without Stop Words =======")
                     print()
                     print("*** Remove Stopwords With nltk *** ")
                     print()
                     nltk.download('stopwords')
                     stop_words = set(stopwords.words('english'))
                     print("*** View a few words from the set")
                     list(stop_words)[0:10]
                     words_in_tweet[0]
                     print("*** Remove stop words from each tweet list of words")
                     tweets_nsw = [[word for word in tweet_words if not word in stop_words]
                     for tweet_words in words_in_tweet]
                     tweets_nsw[0]
                     all_words_nsw = list(itertools.chain(*tweets_nsw))
                     counts_nsw = collections.Counter(all_words_nsw)
                     counts_nsw.most_common(15)
                     clean_tweets_nsw = pd.DataFrame(counts_nsw.most_common(15),
                                 columns=['words', 'count'])
                     fig, ax = plt.subplots(figsize=(8, 8))
                     # Plot horizontal bar graph
                     clean_tweets_nsw.sort_values(by='count').plot.barh(x='words',
                                                 y='count',
                                                 ax=ax,
                                                 color="purple")
                     ax.set_title("Common Words Found in Tweets (Without Stop Words)")
                     plt.show()
                     print()
                     print("--------------------------------------------------------")
                     print()
                     print("*** Menu modele of  graph *** ")
                     print()
                     print(" 0.Exit")
                     print(" 1. Including All Words")
                     print(" 2. Without Stop Words")
                     print(" 3. Without Stop or Collection Words")
                     m1=int(input())
                     print()
                    #time.sleep(t)
                if m1==3:
                    
                     print("======= Without Stop or Collection Words =======")
                     print()
                     print("*** Remove Collection Words****")
                     tweets_nsw_nc = [[w for w in word if not w in tweets_nsw]
                     for word in tweets_nsw]
                     tweets_nsw[0]
                     tweets_nsw_nc[0]
                     # Flatten list of words in clean tweets
                     all_words_nsw_nc = list(itertools.chain(*tweets_nsw_nc))
                     # Create counter of words in clean tweets
                     counts_nsw_nc = collections.Counter(all_words_nsw_nc)
                     counts_nsw_nc.most_common(15)
                     len(counts_nsw_nc)
                     clean_tweets_ncw = pd.DataFrame(counts_nsw_nc.most_common(15),
                                                     columns=['words', 'count'])
                     clean_tweets_ncw.head()
                     fig, ax = plt.subplots(figsize=(8, 8))
                     print("*******Common Words Found in Tweets (Without Stop or Collection Words)*****")
                     print()
                     clean_tweets_ncw.sort_values(by='count').plot.barh(x='words',
                                                 y='count',
                                                 ax=ax,
                                                 color="purple")
                     ax.set_title("Common Words Found in Tweets (Without Stop or Collection Words)")
                     plt.show()
                     print()
                     print("--------------------------------------------------------")
                     print()
                     print("*** Menu modele of  graph *** ")
                     print()
                     print(" 0.Exit")
                     print(" 1. Including All Words")
                     print(" 2. Without Stop Words")
                     print(" 3. Without Stop or Collection Words")
                     m1=int(input())
                     print()
                    #time.sleep(t)
                    
            if(m==4):
                print("=========== Analyze Word Frequency ==================")
                print()
                tweets = tw.Cursor(api.search,
                                   q=search_words,
                                   lang=lg,
                                   since=date).items(nbr)
                def remove_url(txt):
                    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
                # Remove URLs
                tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]
                # Create a sublist of lower case words for each tweet
                words_in_tweet = [tweet.lower().split() for tweet in tweets_no_urls]# Download stopwords
                nltk.download('stopwords')
                stop_words = set(stopwords.words('english'))
                # Remove stop words from each tweet list of words
                tweets_nsw = [[word for word in tweet_words if not word in stop_words]
                for tweet_words in words_in_tweet]
                # Remove collection words
                collection_words = ['climatechange', 'climate', 'change']
                tweets_nsw_nc = [[w for w in word if not w in collection_words]
                for word in tweets_nsw]
                # Create list of lists containing bigrams in tweets
                terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw_nc]
                # View bigrams for the first tweet
                terms_bigram[0]
                # Original tweet without URLs
                tweets_no_urls[0]
                # Clean tweet 
                tweets_nsw_nc[0]
                # Flatten list of bigrams in clean tweets
                bigrams = list(itertools.chain(*terms_bigram))
                # Create counter of words in clean bigrams
                bigram_counts = collections.Counter(bigrams)
                bigram_counts.most_common(20)
                bigram_df = pd.DataFrame(bigram_counts.most_common(20),
                                         columns=['bigram', 'count'])
                bigram_df
                print(bigram_df)
                # Create dictionary of bigrams and their counts
                d = bigram_df.set_index('bigram').T.to_dict('records')
                # Create network plot 
                G = nx.Graph()
                
                # Create connections between nodes
                for k, v in d[0].items():
                    G.add_edge(k[0], k[1], weight=(v * 10))
                
                G.add_node("china", weight=100)
                fig, ax = plt.subplots(figsize=(10, 8))
                
                pos = nx.spring_layout(G, k=1)
                
                # Plot networks
                nx.draw_networkx(G, pos,
                                 font_size=16,
                                 width=3,
                                 edge_color='grey',
                                 node_color='purple',
                                 with_labels = False,
                                 ax=ax)
                
                # Create offset labels
                for key, value in pos.items():
                    x, y = value[0]+.135, value[1]+.045
                    ax.text(x, y,
                            s=key,
                            bbox=dict(facecolor='red', alpha=0.25),
                            horizontalalignment='center', fontsize=13)
                    
                plt.show()
                print()
                print("-------------------------------------")
                print("===> This plot displays the networks of co-occurring words in tweets on climate change.")
                print()
                print()
                print("--------------------------------------------------------")
                print("Menu principale")
                print("-----------------")
                print("       * 0. Exit")
                print("       * 1. Analyze Word Frequency")
                print("       * 2. Nettoyage de data")
                print("       * 3. Calculate and Plot Word Frequency")
                print("       * 4. Analyze Word Frequency")
                print("       * 5. Analyze Sentiments")
                print("       * 6. Visualization and basic statistics")
                #Analyze Sentiments
                m=int(input())
                print()
                print()
            if (m==5):
                print("=========== Analyze Word Frequency ==================")
                print()
                warnings.filterwarnings("ignore")
                sns.set(font_scale=1.5)
                sns.set_style("whitegrid")
                print()
                def remove_url(txt):
                    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())
                # Create a custom search term and define the number of tweets
                tweets = tw.Cursor(api.search,
                                   q=search_words,
                                   lang=lg,
                                   since=date).items(nbr)
                # Remove URLs
                tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]
                print("*** Analyze Sentiments in Tweets ***")
                print()
                print("* Create textblob objects of the tweets")
                print()
                sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_urls]
                sentiment_objects[0].polarity, sentiment_objects[0]
                print("* Create list of polarity valuesx and tweet text")
                print()
                sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]
                sentiment_values[0]
                # Create dataframe containing the polarity value and tweet text
                sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])
                sentiment_df.head()
                print(sentiment_df)
                print()
                print("==>  These polarity values can be plotted in a histogram, which can help to highlight in the overall sentiment (i.e. more positivity or negativity) toward the subject.")
                print()
                print()
                print("***** Sentiments from Tweets on Climate Change ***** ")
                print("---------------------------------------------------")
                print()
                fig, ax = plt.subplots(figsize=(8, 6))
                # Plot histogram of the polarity values
                sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
                                  ax=ax,
                                  color="purple")
                plt.title("Sentiments from Tweets on Climate Change")
                plt.show()
                print()
                print("---------------------------------------------------")
                print("==> This plot displays a histogram of polarity values for tweets on climate change.")
                print()
                print()
                print("--------------------------------------------------------")
                print("Menu principale")
                print("-----------------")
                print("       * 0. Exit")
                print("       * 1. Analyze Word Frequency")
                print("       * 2. Nettoyage de data")
                print("       * 3. Calculate and Plot Word Frequency")
                print("       * 4. Analyze Word Frequency")
                print("       * 5. Analyze Sentiments")
                print("       * 6. Visualization and basic statistics")
                #Analyze Sentiments
                m=int(input())
                print()
                print()
            if m==6:
                print("====Visualization and basic statistics=======")
                print()
                def twitter_setup():    
                    auth = tw.OAuthHandler(consumer_key, consumer_secret)
                    auth.set_access_token(access_token,access_token_secret)
                    # Return API with authentication:
                    api = tw.API(auth)
                    return api
                extractor = twitter_setup()
                # We create a tweet list as follows:
                tweets = extractor.user_timeline(screen_name=search_words, count=nbr)
                print("****Number of tweets : {}.\n".format(len(tweets)))
                print()
                # We create a pandas dataframe as follows:
                data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
                # We display the first 10 elements of the dataframe:
                print("**** dataframe as follows ***")
                print()
                display(data.head(nbr))
                # We add relevant data:
                data['len']  = np.array([len(tweet.text) for tweet in tweets])
                data['ID']   = np.array([tweet.id for tweet in tweets])
                data['Date'] = np.array([tweet.created_at for tweet in tweets])
                data['Source'] = np.array([tweet.source for tweet in tweets])
                data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
                data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])
                # Display of first 10 elements from dataframe:
                display(data.head(nbr))
                print()
                print()
                print("---------------------------------------------------")
                print("==== Visualization and basic statistics ========== ")
                print("---------------------------------------------------")
                print()
                # We extract the mean of lenghts:
                mean = np.mean(data['len'])
                print(" *** The lenght's average in tweets : {}".format(mean))
                print()
                print("*** the tweet with more FAVs and more RTs *** ")
                print()
                fav_max = np.max(data['Likes'])
                rt_max  = np.max(data['RTs'])
                fav = data[data.Likes == fav_max].index[0]
                rt  = data[data.RTs == rt_max].index[0]
                print("Max FAVs:")
                print("------------------------------------------------------")
                print("* The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
                print()
                print("* Number of likes: {}".format(fav_max))
                print()
                print("{} characters.\n".format(data['len'][fav]))
                print()
                print("------------------------------------------------------")
                print()
                print("# Max RTs:")
                print("------------------------------------------------------")
                print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
                print()
                print("Number of retweets: {}".format(rt_max))
                print()
                print("{} characters.\n".format(data['len'][rt]))
                print()     
                print("------------------------------------------------------")
                print()
                print()
                #time series for data
                print("*** time series for data *** ")
                print()
                print(" * Plot a time series as follows:***")
                tlen = pd.Series(data=data['len'].values, index=data['Date'])
                tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
                tret = pd.Series(data=data['RTs'].values, index=data['Date'])
                # Lenghts along time:
                print()
                print("------------------------------------------------------")
                print()
                print("===== Likes vs retweets visualization===")
                print()
                tlen.plot(figsize=(16,4), color='r');
                print("****plot the likes versus the retweets in the same chart:****")
                tfav.plot(figsize=(16,4), label="Likes", legend=True)
                tret.plot(figsize=(16,4), label="Retweets", legend=True);
                print()
                print("------------------------------------------------------")
                print()
                print("*** all possible sources ***")
                print()
                sources = []
                for source in data['Source']:
                    if source not in sources:
                        sources.append(source)
                for source in sources:
                    print("* {}".format(source))
                print()
                print("------------------------------------------------------")
                print()
                print("create a numpy vector mapped to labels")
                print()
                percent = np.zeros(len(sources))
                
                for source in data['Source']:
                    for index in range(len(sources)):
                        if source == sources[index]:
                            percent[index] += 1
                            pass
                
                percent /= 100
                pie_chart = pd.Series(percent, index=sources, name='Sources')
                pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6));
                print()
                print("------------------------------------------------------")
                print()
                print("============Sentiment analysis=======")
                print()
                def clean_tweet(tweet):
                    '''
                    Utility function to clean the text in a tweet by removing
                    links and special characters using regex.
                    '''
                    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
                
                def analize_sentiment(tweet):
                    '''
                    Utility function to classify the polarity of a tweet
                    using textblob.
                    '''
                    analysis = TextBlob(clean_tweet(tweet))
                    if analysis.sentiment.polarity > 0:
                        return 1
                    elif analysis.sentiment.polarity == 0:
                        return 0
                    else:
                        return -1
                    
                print()
                print("------------------------------------------------------")
                print()
                # We create a column with the result of the analysis:
                data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])
                #****************************************************************
                
                # We display the updated dataframe with the new column:
                display(data.head(nbr))
                print()
                print()
                print("================Analyzing the results==================")
                print()
                print("------------------------------------------------------")
                print()
                pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
                neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
                neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]
                print()
                print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
                print()
                print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
                print()
                print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))
                print()
                print("--------------------------------------------------------")
                print("Menu principale")
                print("-----------------")
                print("       * 0. Exit")
                print("       * 1. Analyze Word Frequency")
                print("       * 2. Nettoyage de data")
                print("       * 3. Calculate and Plot Word Frequency")
                print("       * 4. Analyze Word Frequency")
                print("       * 5. Analyze Sentiments")
                print("       * 6. Visualization and basic statistics")
                #Analyze Sentiments
                m=int(input())
                print()
                print()