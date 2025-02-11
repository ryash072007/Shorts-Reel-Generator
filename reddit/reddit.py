import requests
import random

SUBREDDIT_NAMES = ["story", "prorevenge", "pettyrevenge", "maliciouscompliance"]


def get_posts_content(subreddit_name, limit=1):
    url = f"https://www.reddit.com/r/{subreddit_name}/best.json?sort=top&t=day&limit={limit}"
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


if __name__ == "__main__":
    num_posts = 1
    posts = get_posts_content(random.choice(SUBREDDIT_NAMES), num_posts)
    if posts:
        print(posts)
    else:
        print("Failed to retrieve the posts.")
