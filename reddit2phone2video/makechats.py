"""
Simple utility to generate chat conversations from Reddit content
"""

import json
import logging
import os
import random
import requests
from typing import Dict, Any, List, Optional, Tuple
from groq import Groq
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("makechats")

class RedditClient:
    """Client for fetching posts from Reddit."""
    
    _session = None
    
    @classmethod
    def get_session(cls) -> requests.Session:
        """Get or create a reusable HTTP session."""
        if cls._session is None:
            cls._session = requests.Session()
            cls._session.headers.update({
                "User-Agent": "Chrome/133.0.6943.98",
            })
        return cls._session
    
    @staticmethod
    def get_posts(subreddit: str = "pettyrevenge", post_type: str = "hot", time_frame: str = "day") -> Dict[str, Any]:
        """Fetch posts from Reddit using a persistent session."""
        url = f"https://www.reddit.com/r/{subreddit}/{post_type}/.json"
        
        if post_type == "top":
            url += f"?t={time_frame}"
        
        logger.info(f"Fetching posts from r/{subreddit} ({post_type}, {time_frame})")
        session = RedditClient.get_session()
        response = session.get(url)
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_random_post(subreddit: str = "WritingPrompts", min_upvotes: int = 6000) -> Dict[str, Any]:
        """Fetch a random post from Reddit with at least the specified number of upvotes."""
        posts = RedditClient.get_posts(subreddit, post_type="top", time_frame="week")
        eligible_posts = [post for post in posts["data"]["children"] 
                          if post["data"].get("ups", 0) >= min_upvotes]
        
        if not eligible_posts:
            logger.warning(f"No posts found with at least {min_upvotes} upvotes in r/{subreddit}")
            # Try with a wider time frame
            posts = RedditClient.get_posts(subreddit, post_type="top", time_frame="month")
            eligible_posts = [post for post in posts["data"]["children"] 
                              if post["data"].get("ups", 0) >= min_upvotes]
            
            if not eligible_posts:
                logger.warning(f"Still no posts found with {min_upvotes} upvotes, using best available")
                # Sort by upvotes and take the highest ones
                sorted_posts = sorted(posts["data"]["children"], 
                                     key=lambda post: post["data"].get("ups", 0), 
                                     reverse=True)
                return sorted_posts[0] if sorted_posts else None
        
        logger.info(f"Found {len(eligible_posts)} posts with at least {min_upvotes} upvotes")
        return random.choice(eligible_posts)


class AIClient:
    """Simple client for AI text generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the AI client with API key."""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required as environment variable")
        
        self.client = Groq(api_key=self.api_key)
    
    @lru_cache(maxsize=10)
    def get_json_response(self, prompt: str, model: str = "deepseek-r1-distill-qwen-32b") -> Dict[str, Any]:
        """Get JSON response from AI model with caching."""
        logger.info("Sending request to Groq API for JSON response")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            result = json.loads(chat_completion.choices[0].message.content)
            logger.info("Successfully received JSON response")
            return result
        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            raise

    @lru_cache(maxsize=10)
    def get_text_response(self, prompt: str, model: str = "deepseek-r1-distill-qwen-32b") -> str:
        """Get text response from AI model with caching."""
        logger.info("Sending request to Groq API for text response")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.7,
            )
            result = chat_completion.choices[0].message.content
            logger.info("Successfully received text response")
            return result
        except Exception as e:
            logger.error(f"Error in API call: {str(e)}")
            raise


def generate_chat_from_story(story: str) -> Dict[str, Any]:
    """Generate a chat conversation from a story."""
    ai_client = AIClient()
    
    prompt = \
        f"""Turn the following story into a dramatic text conversation
        that could go viral on social media. It should have very dramatic
        hook. Do not leave any part of the story. It should make sense as a conversation between two people.
        Give the speakers realistic names.
        Return in JSON format with 'title' and 'messages' fields.
        Each message should have 'sender' and 'text' fields. Story: {story}"""
    
    
    return ai_client.get_json_response(prompt)


def main():
    """Main entry point."""
    # Get a random post from WritingPrompts with at least 6000 upvotes
    try:
        post = RedditClient.get_random_post("PettyRevenge", min_upvotes=6000)
        if not post:
            print("No suitable posts found. Try again later or with different parameters.")
            return
            
        post_data = post["data"]
        story = post_data["title"] + "\n\n" + post_data.get("selftext", "")
        
        print(f"\n=== ORIGINAL STORY ===\n{story}\n")
        print(f"Upvotes: {post_data.get('ups', 'Unknown')}\n")
        
        # Generate chat conversation
        chat_data = generate_chat_from_story(story)
        
        print("\n=== CHAT CONVERSATION ===")
        print(f"Title: {chat_data.get('title')}\n")
        
        for msg in chat_data.get('messages', []):
            print(f"{msg.get('sender')}: {msg.get('text')}")
            
        # Save to file
        output_file = "reddit2phone2video/chat_conversation.json"
        with open(output_file, "w") as f:
            json.dump(chat_data, f, indent=2)
        print(f"\nSaved conversation to {output_file}")

        return chat_data
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
