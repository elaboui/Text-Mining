from tweepy import Stream
from tweepy import OAuthHandler
from tweepy import StreamListener



ACCESS_TOKEN = "1223301958754357250-CZ1TU2ankbs76fyYDVoitgL7TFwmKy"
ACCESS_TOKEN_SECRET ="gppuJoR7O3r9XUVy2iv72TLBCbIGlmpW3JfbM1RxA2FP3"
CONSUMER_KEY = "6aafmRHOMinQmOUxjzGG2AzoE"
CONSUMER_SECRET = "ExvFnoOb87IVwr4MjY7Jf9ZeecWbaTdeSs8RjYTTHquxUEEAa0"
# OAuth process
auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

data = list()
# listener that handles streaming data
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


def main():
    twitterStream = Stream(auth, listener())
    twitterStream.filter(locations=[-130.78125, -31.3536369415, 140.625, 63.8600358954])
