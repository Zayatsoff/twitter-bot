import tweepy
import json

# Open JSON file
with open("key.JSON") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

# Authenticate to Twitter
auth = tweepy.OAuthHandler(jsonObject["api_key"], jsonObject["api_key_secret"])
auth.set_access_token(
    jsonObject["access_token"], jsonObject["access_token_secret"],
)
api = tweepy.API(auth)

