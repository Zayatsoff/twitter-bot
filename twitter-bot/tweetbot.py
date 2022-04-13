from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import tweepy
import json


class TweetBot:
    def __init__(
        self,
        model_name="mahaamami/distilgpt2-finetuned-wikitext2",
        min_length=100,
        temperature=0.5,
    ):
        self.model_name = model_name
        self.min_length = min_length
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
            self.device
        )
        self.gen = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            device=0,
        )
        torch.save(self.model, f"{self.model_name}.pt")

    def generate(self, prompt):
        output = self.gen(
            prompt,
            do_sample=True,
            min_length=self.min_length,
            temperature=self.temperature,
        )
        split_output = output[0]["generated_text"].split(".", 1)[0] + "."
        return split_output

    def create_tweet(self, prompt="Fun fact:", key_loc="key.JSON"):
        tweet = self.generate(prompt)
        api = self.auth_twitter(key_loc)
        api.update_status(status=tweet)
        print(f"Generated text: {tweet}")

    def auth_twitter(self, key_loc):
        # Open JSON file
        with open(key_loc) as jsonFile:
            jsonObject = json.load(jsonFile)
            jsonFile.close()

        # Authenticate to Twitter
        auth = tweepy.OAuthHandler(jsonObject["api_key"], jsonObject["api_key_secret"])
        auth.set_access_token(
            jsonObject["access_token"], jsonObject["access_token_secret"],
        )
        api = tweepy.API(auth, wait_on_rate_limit=True)
        return api

