import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
from tqdm import tqdm
import praw
import datetime
import json
from collections import Counter
from wordcloud import WordCloud
import re

class YouTubeRedditResearcher:
    def __init__(self, youtube_api_key, reddit_client_id=None, reddit_client_secret=None, 
                 reddit_user_agent=None):
        """Initialize the research tool with API credentials."""
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        
        # Initialize Reddit API if credentials are provided
        self.reddit = None
        if reddit_client_id and reddit_client_secret and reddit_user_agent:
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=reddit_user_agent
            )
        
        # Storage for collected data
        self.shorts_data = []
        self.trending_reddit_data = {}
        self.analysis_results = {}
        
    def search_reddit_shorts(self, query_terms=None, max_results=100):
        """
        Search for YouTube Shorts containing Reddit content.
        
        Args:
            query_terms (list): Additional search terms beyond 'reddit'
            max_results (int): Maximum number of results to return
        """
        if query_terms is None:
            query_terms = ['reddit']
        else:
            query_terms.append('reddit')
            
        base_query = ' '.join(query_terms)
        search_queries = [
            f"{base_query} shorts",
            f"{base_query} short",
            f"{base_query} #shorts",
        ]
        
        for query in search_queries:
            print(f"Searching for: {query}")
            
            next_page_token = None
            results_count = 0
            
            while results_count < max_results:
                # Perform search
                search_response = self.youtube.search().list(
                    q=query,
                    part='id,snippet',
                    maxResults=min(50, max_results - results_count),
                    type='video',
                    videoDefinition='high',
                    videoDuration='short',
                    pageToken=next_page_token
                ).execute()
                
                video_ids = [item['id']['videoId'] for item in search_response['items']]
                
                # Get video statistics
                if video_ids:
                    videos_response = self.youtube.videos().list(
                        part='statistics,snippet,contentDetails',
                        id=','.join(video_ids)
                    ).execute()
                    
                    # Process each video
                    for video in videos_response['items']:
                        # Check if it's actually a short (vertical video)
                        if 'shorts' in video['snippet'].get('title', '').lower() or \
                           'shorts' in video['snippet'].get('tags', []) or \
                           'shorts' in video['snippet'].get('description', '').lower():
                            
                            # Extract relevant data
                            video_data = {
                                'id': video['id'],
                                'title': video['snippet']['title'],
                                'channel_id': video['snippet']['channelId'],
                                'channel_title': video['snippet']['channelTitle'],
                                'publish_date': video['snippet']['publishedAt'],
                                'description': video['snippet']['description'],
                                'tags': video['snippet'].get('tags', []),
                                'view_count': int(video['statistics'].get('viewCount', 0)),
                                'like_count': int(video['statistics'].get('likeCount', 0)),
                                'comment_count': int(video['statistics'].get('commentCount', 0)),
                                'duration': video['contentDetails']['duration'],
                                'search_query': query
                            }
                            
                            self.shorts_data.append(video_data)
                            results_count += 1
                
                # Check if there are more results
                next_page_token = search_response.get('nextPageToken')
                if not next_page_token or results_count >= max_results:
                    break
        
        print(f"Collected data on {len(self.shorts_data)} YouTube Shorts")
        return self.shorts_data
    
    def analyze_trending_subreddits(self):
        """Analyze which subreddits are most featured in successful YouTube Shorts."""
        if not self.reddit:
            print("Reddit API credentials not provided. Skipping Reddit analysis.")
            return {}
            
        # Extract subreddit mentions from video titles and descriptions
        subreddit_pattern = r'r\/([A-Za-z0-9_]+)'
        
        subreddit_mentions = []
        for video in self.shorts_data:
            # Look for subreddit mentions in title
            title_mentions = re.findall(subreddit_pattern, video['title'])
            subreddit_mentions.extend([(sub.lower(), video['view_count']) for sub in title_mentions])
            
            # Look for subreddit mentions in description
            desc_mentions = re.findall(subreddit_pattern, video['description'])
            subreddit_mentions.extend([(sub.lower(), video['view_count']) for sub in desc_mentions])
        
        # Aggregate mentions
        subreddit_stats = {}
        for subreddit, views in subreddit_mentions:
            if subreddit not in subreddit_stats:
                subreddit_stats[subreddit] = {
                    'mention_count': 0,
                    'total_views': 0,
                    'avg_views': 0
                }
            
            subreddit_stats[subreddit]['mention_count'] += 1
            subreddit_stats[subreddit]['total_views'] += views
        
        # Calculate average views
        for subreddit in subreddit_stats:
            subreddit_stats[subreddit]['avg_views'] = (
                subreddit_stats[subreddit]['total_views'] / 
                subreddit_stats[subreddit]['mention_count']
            )
        
        # Sort by mention count
        sorted_subreddits = sorted(
            subreddit_stats.items(),
            key=lambda x: x[1]['mention_count'],
            reverse=True
        )
        
        self.trending_reddit_data = dict(sorted_subreddits)
        return self.trending_reddit_data
    
    def analyze_content_trends(self):
        """Analyze common themes, formats, and patterns in successful shorts."""
        if not self.shorts_data:
            print("No data to analyze. Run search_reddit_shorts first.")
            return {}
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.shorts_data)
        
        # Define high-performing videos (e.g., top 25% by views)
        view_threshold = df['view_count'].quantile(0.75)
        high_performing = df[df['view_count'] >= view_threshold]
        
        # Basic statistics
        stats = {
            'total_videos': len(df),
            'high_performing_count': len(high_performing),
            'avg_views': df['view_count'].mean(),
            'median_views': df['view_count'].median(),
            'high_performer_avg_views': high_performing['view_count'].mean(),
            'engagement_ratio': (df['like_count'].sum() + df['comment_count'].sum()) / df['view_count'].sum()
        }
        
        # Content analysis - word frequency in titles
        all_words = ' '.join(df['title']).lower()
        all_words = re.sub(r'[^\w\s]', '', all_words)
        word_counts = Counter(all_words.split())
        
        high_words = ' '.join(high_performing['title']).lower()
        high_words = re.sub(r'[^\w\s]', '', high_words)
        high_word_counts = Counter(high_words.split())
        
        # Tag analysis
        all_tags = []
        for tags in df['tags']:
            if isinstance(tags, list):
                all_tags.extend(tags)
        
        tag_counts = Counter(all_tags)
        
        # Combine results
        analysis = {
            'stats': stats,
            'common_words': dict(word_counts.most_common(30)),
            'high_performing_words': dict(high_word_counts.most_common(30)),
            'common_tags': dict(tag_counts.most_common(30)),
        }
        
        self.analysis_results = analysis
        return analysis
    
    def identify_untapped_niches(self):
        """Identify potential untapped niches based on the analysis."""
        if not self.analysis_results or not self.trending_reddit_data:
            print("Run analyze_content_trends and analyze_trending_subreddits first.")
            return []
            
        # Look for subreddits with high average views but low mention count
        potential_niches = []
        
        for subreddit, stats in self.trending_reddit_data.items():
            if stats['avg_views'] > self.analysis_results['stats']['avg_views'] * 1.5 and \
               stats['mention_count'] < 5:
                potential_niches.append({
                    'subreddit': subreddit,
                    'avg_views': stats['avg_views'],
                    'mention_count': stats['mention_count'],
                    'potential': stats['avg_views'] / (stats['mention_count'] + 1)
                })
        
        # Sort by potential
        potential_niches.sort(key=lambda x: x['potential'], reverse=True)
        
        return potential_niches
    
    def plot_top_performers(self, top_n=10):
        """Plot the top performing videos by views."""
        if not self.shorts_data:
            print("No data available. Run search_reddit_shorts first.")
            return
        
        df = pd.DataFrame(self.shorts_data)
        top_videos = df.nlargest(top_n, 'view_count')
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='view_count', y='title', data=top_videos)
        plt.title(f'Top {top_n} Performing Reddit-Based YouTube Shorts')
        plt.xlabel('View Count')
        plt.ylabel('Video Title')
        plt.tight_layout()
        plt.show()
    
    def visualize_content_trends(self):
        """Create visualizations of content trends."""
        if not self.analysis_results:
            print("No analysis results available. Run analyze_content_trends first.")
            return
        
        # Word cloud of high-performing video titles
        plt.figure(figsize=(10, 6))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            self.analysis_results['high_performing_words']
        )
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Common Words in High-Performing Videos')
        plt.tight_layout()
        plt.show()
        
        # Bar chart of top subreddits
        if self.trending_reddit_data:
            top_subs = {k: v['mention_count'] for k, v in list(self.trending_reddit_data.items())[:10]}
            plt.figure(figsize=(12, 6))
            plt.bar(top_subs.keys(), top_subs.values())
            plt.title('Top 10 Subreddits Featured in YouTube Shorts')
            plt.xlabel('Subreddit')
            plt.ylabel('Mention Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    
    def save_research(self, filename='youtube_reddit_research.json'):
        """Save the research data to a file."""
        research_data = {
            'shorts_data': self.shorts_data,
            'trending_reddit_data': self.trending_reddit_data,
            'analysis_results': self.analysis_results,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(research_data, f, indent=2)
            
        print(f"Research data saved to {filename}")
    
    def generate_report(self):
        """Generate a comprehensive research report with recommendations."""
        if not self.analysis_results:
            print("No analysis results available. Run analyze_content_trends first.")
            return ""
            
        untapped_niches = self.identify_untapped_niches()
        
        report = []
        report.append("# YouTube Shorts Research Report: Reddit Content")
        report.append("\n## Overview")
        report.append(f"Total videos analyzed: {self.analysis_results['stats']['total_videos']}")
        report.append(f"Average views: {int(self.analysis_results['stats']['avg_views']):,}")
        report.append(f"Median views: {int(self.analysis_results['stats']['median_views']):,}")
        
        report.append("\n## Top Performing Content")
        report.append("### Most Common Words in High-Performing Videos:")
        top_words = sorted(self.analysis_results['high_performing_words'].items(), 
                           key=lambda x: x[1], reverse=True)[:15]
        for word, count in top_words:
            if len(word) > 3:  # Filter out common short words
                report.append(f"- {word}: {count}")
        
        report.append("\n### Top Subreddits Featured:")
        if self.trending_reddit_data:
            for i, (subreddit, stats) in enumerate(list(self.trending_reddit_data.items())[:10]):
                report.append(f"{i+1}. r/{subreddit}: {stats['mention_count']} mentions, "
                              f"{int(stats['avg_views']):,} avg views")
        
        report.append("\n## Untapped Opportunities")
        if untapped_niches:
            report.append("Subreddits with high potential but lower competition:")
            for i, niche in enumerate(untapped_niches[:10]):
                report.append(f"{i+1}. r/{niche['subreddit']}: {int(niche['avg_views']):,} avg views, "
                              f"only {niche['mention_count']} mentions")
        
        report.append("\n## Content Recommendations")
        report.append("1. Focus on emerging trends in these subreddits with high engagement potential")
        report.append("2. Create shorts that combine multiple subreddit content for broader appeal")
        report.append("3. Use these high-performing title keywords: " + 
                     ", ".join([word for word, _ in top_words[:10] if len(word) > 3]))
        report.append("4. Target the 'sweet spot' duration of 30-45 seconds for maximum retention")
        
        report.append("\n## AI Content Creation Strategy")
        report.append("1. Use AI to monitor trending posts in the identified subreddits in real-time")
        report.append("2. Implement voice synthesis that matches the emotional tone of the content")
        report.append("3. Create templates for different content types (reactions, stories, jokes)")
        report.append("4. Develop a batching system to produce variations of successful formats")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Replace with your actual API keys
    researcher = YouTubeRedditResearcher(
        youtube_api_key="AIzaSyD8iqanwISiYaAk6iKUEJd0-FWWhxzqlN8",
        reddit_client_id="AMnxBkvb3age0TFS1e3T9w",
        reddit_client_secret="2YLX8fiLQvyma0gxvdWyU2q9Cr90oA",
        reddit_user_agent="YTResearch/1.0"
    )
    
    # Search for Reddit shorts
    researcher.search_reddit_shorts(
        query_terms=["reddit", "AITA", "relationship", "confession"],
        max_results=100
    )
    
    # Analyze the data
    researcher.analyze_trending_subreddits()
    researcher.analyze_content_trends()
    
    # Generate insights
    untapped_niches = researcher.identify_untapped_niches()
    
    # Visualize the results
    researcher.plot_top_performers()
    researcher.visualize_content_trends()
    
    # Generate and print report
    report = researcher.generate_report()
    print(report)
    
    # Save the research
    researcher.save_research()
