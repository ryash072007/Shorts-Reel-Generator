import requests
from groq import Groq
import os, time

SUBREDDITS = ["memes", "MinecraftMemes", "wholesomememes"]

MIN_UPVOTES = 2000

GROQCLIENT = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def get_text_reply(prompt, _model="qwen-2.5-32b"):
    print("[GROQ] Sending request to Groq API...")
    try:
        chat_completion = GROQCLIENT.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=_model,
        )
        print("[GROQ] Received response from Groq API")
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"[GROQ] Error in API call: {str(e)}")

        if "Please try again in " in str(e):
            wt = str(e).split("Please try again in ")[1].split(". ")[0].split("m")
            wait_time = float(wt[0]) * 60 + float(wt[1].split("s")[0])
            print(f"[DEBUG] Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
        return get_text_reply(prompt)


def get_image_reply(text, image_url, _model="llama-3.2-11b-vision-preview"):
    print("[GROQ] Sending request to Groq API...")
    try:
        image_completion = GROQCLIENT.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                            + " You have to return the result in json format.",
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            model=_model,
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        print("[GROQ] Received response from Groq API")

        message = image_completion.choices[0].message.content


        return message
    except Exception as e:
        print(f"[GROQ] Error in API call: {str(e)}")

        if "Please try again in " in str(e):
            wt = str(e).split("Please try again in ")[1].split(". ")[0].split("m")
            wait_time = float(wt[0]) * 60 + float(wt[1].split("s")[0])
            print(f"[DEBUG] Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
        return get_image_reply(text, image_url)


def get_reddit_posts(subreddit="memes", type = "hot"):
    url = f"https://www.reddit.com/r/{subreddit}/{type}/.json"
    headers = {
        "User-Agent": "Chrome/133.0.6943.98",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_meme_urls(posts: list):
    meme_urls: list = []
    for post in posts["data"]["children"]:
        if post["data"].get("url_overridden_by_dest") is None:
            continue
        if post["data"]["url_overridden_by_dest"].endswith((".jpeg", ".png")):
            if post["data"]["ups"] > MIN_UPVOTES:
                meme_urls.append(post["data"]["url_overridden_by_dest"])
    return meme_urls


if __name__ == "__main__":
    posts = get_reddit_posts(subreddit=SUBREDDITS[1])
    meme_urls = get_meme_urls(posts)[0]

    image_reply = get_image_reply(
        """Analyze the provided image and identify the captions that are part of the meme's intended dialogue or message. You should not have to describe the meme format/image as this is a OCR tool. Only the relevant text on the image has to be processed. Ignore any watermarks, credits, or unrelated text. Do not duplicate any captions.  Determine the natural reading order as a human would perceive it. Then, output the captions in JSON format with an ordered list, as follows:
        {
  "reading_order": [
    "First caption",
    "Second caption",
  ]
}
Ensure the order reflects logical reading patterns based on spatial positioning and dialogue structure."
"""
    , meme_urls)

    print(meme_urls)
    print(image_reply)
