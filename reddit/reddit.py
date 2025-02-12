import requests
import random

SUBREDDIT_NAMES = ["story"]#, "prorevenge", "pettyrevenge", "maliciouscompliance"]


def get_posts_content(subreddit_name, limit=1):
    url = f"https://www.reddit.com/r/{subreddit_name}/best.json?sort=top&t=day"#{limit}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        posts = data["data"]["children"]
        return [
            {"title": post["data"]["title"], "content": post["data"]["selftext"]}
            for post in posts
        ]
    else:
        get_posts_content(subreddit_name, limit)

def break_down_text(text, nolines=3):
    lines = text.split(".")
    prompts = []
    for i in range(0, len(lines), nolines):
        prompt = ". ".join(lines[i:i+nolines]).strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def get_post_and_breakdown(nolines = 3):
    posts = get_posts_content(random.choice(SUBREDDIT_NAMES))
    if posts:
        post = posts[0]
        return {
            "post": post,
            "breakdown": break_down_text(post["content"], nolines)
        }
    else:
        return None


if __name__ == "__main__":
    num_posts = 1
    posts = get_posts_content(random.choice(SUBREDDIT_NAMES), num_posts)
    if posts:
        print(posts)
    else:
        print("Failed to retrieve the posts.")
