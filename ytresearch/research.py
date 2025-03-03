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
import requests
from dotenv import load_dotenv
import argparse
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ytresearch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("YouTubeResearcher")

# Load environment variables
load_dotenv()

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class YouTubeRedditResearcher:
    def __init__(self, youtube_api_key=None, reddit_client_id=None, reddit_client_secret=None, 
                 reddit_user_agent=None, groq_api_key=None):
        """Initialize the research tool with API credentials."""
        # Use environment variables if not provided directly
        youtube_api_key = youtube_api_key or os.environ.get('YOUTUBE_API_KEY')
        reddit_client_id = reddit_client_id or os.environ.get('REDDIT_CLIENT_ID')
        reddit_client_secret = reddit_client_secret or os.environ.get('REDDIT_CLIENT_SECRET')
        reddit_user_agent = reddit_user_agent or os.environ.get('REDDIT_USER_AGENT', 'YTResearch/1.0')
        self.groq_api_key = groq_api_key or os.environ.get('GROQ_API_KEY')
        
        if not youtube_api_key:
            raise ValueError("YouTube API key is required. Set YOUTUBE_API_KEY in .env or pass as parameter.")
            
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
        
        # Additional storage for enhanced analysis
        self.channel_growth_data = {}
        self.success_path_analysis = {}
        self.competitive_landscape = {}
        self.content_evolution_data = {}
        
    def search_reddit_shorts(self, query_terms=None, max_results=100):
        """
        Search for YouTube Shorts containing Reddit content.
        
        Args:
            query_terms (list): Additional search terms beyond 'reddit'
            max_results (int): Maximum number of results to return
        """
        try:
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
        except Exception as e:
            logger.error(f"Error in search_reddit_shorts: {e}")
            logger.error(traceback.format_exc())
            print(f"Error searching for shorts: {e}")
            return []

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

    def analyze_channel_growth(self, min_channels=10, max_channels=50):
        """
        Analyze how channels in this niche have grown over time.
        Tracks small channels that became successful and their growth trajectory.
        """
        if not self.shorts_data:
            print("No data to analyze. Run search_reddit_shorts first.")
            return {}
        
        # Extract unique channels from our data
        df = pd.DataFrame(self.shorts_data)
        channels = df[['channel_id', 'channel_title']].drop_duplicates()
        
        print(f"Analyzing growth patterns for {min(len(channels), max_channels)} channels...")
        channel_growth = {}
        
        # For each channel, get their video history to analyze growth
        for idx, (channel_id, channel_title) in enumerate(
            zip(channels['channel_id'].values[:max_channels], 
                channels['channel_title'].values[:max_channels])
        ):
            print(f"Analyzing channel {idx+1}/{min(len(channels), max_channels)}: {channel_title}")
            
            try:
                # Get channel statistics
                channel_response = self.youtube.channels().list(
                    part="snippet,statistics,contentDetails",
                    id=channel_id
                ).execute()
                
                if not channel_response['items']:
                    continue
                    
                channel_info = channel_response['items'][0]
                
                # Get channel creation date
                created_at = channel_info['snippet']['publishedAt']
                subscriber_count = int(channel_info['statistics'].get('subscriberCount', 0))
                view_count = int(channel_info['statistics'].get('viewCount', 0))
                video_count = int(channel_info['statistics'].get('videoCount', 0))
                
                # Calculate age in days
                creation_date = datetime.datetime.strptime(created_at.split('T')[0], '%Y-%m-%d')
                today = datetime.datetime.now()
                channel_age_days = (today - creation_date).days
                
                # Get upload playlist ID
                uploads_playlist_id = channel_info['contentDetails']['relatedPlaylists']['uploads']
                
                # Get video history (most recent 50 videos)
                video_history = []
                next_page_token = None
                total_videos = 0
                
                while total_videos < 50:
                    playlist_response = self.youtube.playlistItems().list(
                        part="snippet,contentDetails",
                        playlistId=uploads_playlist_id,
                        maxResults=50,
                        pageToken=next_page_token
                    ).execute()
                    
                    for item in playlist_response['items']:
                        video_id = item['contentDetails']['videoId']
                        published_at = item['snippet']['publishedAt']
                        title = item['snippet']['title']
                        
                        # Only get stats for shorts
                        if ('shorts' in title.lower() or 
                            'short' in title.lower() or 
                            '#shorts' in title.lower()):
                            
                            try:
                                # Get video stats
                                video_response = self.youtube.videos().list(
                                    part="statistics,contentDetails",
                                    id=video_id
                                ).execute()
                                
                                if video_response['items']:
                                    video_stats = video_response['items'][0]['statistics']
                                    view_count = int(video_stats.get('viewCount', 0))
                                    like_count = int(video_stats.get('likeCount', 0))
                                    comment_count = int(video_stats.get('commentCount', 0))
                                    duration = video_response['items'][0]['contentDetails']['duration']
                                    
                                    video_history.append({
                                        'video_id': video_id,
                                        'published_at': published_at,
                                        'title': title,
                                        'view_count': view_count,
                                        'like_count': like_count,
                                        'comment_count': comment_count,
                                        'duration': duration
                                    })
                            except Exception as e:
                                print(f"Error getting video stats: {e}")
                    
                    total_videos += len(playlist_response['items'])
                    next_page_token = playlist_response.get('nextPageToken')
                    if not next_page_token:
                        break
                
                # Calculate growth metrics
                if video_history:
                    # Sort by publication date
                    video_history.sort(key=lambda x: x['published_at'])
                    
                    # Divide history into quarters to see growth
                    quarters = np.array_split(video_history, 4)
                    quarter_stats = []
                    
                    for i, quarter in enumerate(quarters):
                        if quarter.size > 0:
                            avg_views = np.mean([v['view_count'] for v in quarter])
                            avg_engagement = np.mean([
                                (v['like_count'] + v['comment_count']) / max(v['view_count'], 1) 
                                for v in quarter
                            ])
                            quarter_stats.append({
                                'quarter': i + 1,
                                'avg_views': avg_views,
                                'avg_engagement': avg_engagement,
                                'video_count': len(quarter)
                            })
                    
                    # Calculate growth rate
                    if len(quarter_stats) > 1:
                        first_quarter = quarter_stats[0]['avg_views']
                        last_quarter = quarter_stats[-1]['avg_views']
                        if first_quarter > 0:
                            view_growth_rate = (last_quarter / first_quarter) - 1
                        else:
                            view_growth_rate = float('inf')
                    else:
                        view_growth_rate = 0
                    
                    # Calculate daily subscriber growth rate
                    subs_per_day = subscriber_count / max(channel_age_days, 1)
                    
                    # Determine success tier based on subscriber count and growth
                    if subscriber_count >= 1000000:
                        success_tier = "Mega"
                    elif subscriber_count >= 100000:
                        success_tier = "Large"
                    elif subscriber_count >= 10000:
                        success_tier = "Medium"
                    elif subscriber_count >= 1000:
                        success_tier = "Small"
                    else:
                        success_tier = "Micro"
                    
                    # Store growth data
                    channel_growth[channel_id] = {
                        'channel_title': channel_title,
                        'created_at': created_at,
                        'channel_age_days': channel_age_days,
                        'subscriber_count': subscriber_count,
                        'total_view_count': view_count,
                        'video_count': video_count,
                        'subscriber_growth_per_day': subs_per_day,
                        'view_growth_rate': view_growth_rate,
                        'success_tier': success_tier,
                        'quarter_stats': quarter_stats,
                        'recent_videos': video_history[:10]  # Store 10 most recent videos
                    }
                    
            except Exception as e:
                print(f"Error analyzing channel {channel_title}: {e}")
        
        self.channel_growth_data = channel_growth
        
        # Find success stories (small channels that grew quickly)
        success_stories = sorted(
            [(cid, data) for cid, data in channel_growth.items() 
             if data['channel_age_days'] <= 365 and data['subscriber_count'] >= 10000],
            key=lambda x: x[1]['subscriber_count'],
            reverse=True
        )
        
        if success_stories:
            print("\nSuccess stories - Channels that grew quickly:")
            for cid, data in success_stories[:5]:
                print(f"- {data['channel_title']}: {data['subscriber_count']:,} subscribers "
                      f"in {data['channel_age_days']} days")
                      
        return channel_growth
    
    def analyze_success_patterns(self):
        """Extract common patterns from successful channels in the niche."""
        if not self.channel_growth_data:
            print("No channel growth data. Run analyze_channel_growth first.")
            return {}
            
        # Define success tiers for analysis
        success_tiers = {
            'Mega': [],   # 1M+
            'Large': [],  # 100K-1M
            'Medium': [], # 10K-100K
            'Small': [],  # 1K-10K
            'Micro': []   # <1K
        }
        
        # Group channels by success tier
        for channel_id, data in self.channel_growth_data.items():
            tier = data['success_tier']
            success_tiers[tier].append((channel_id, data))
            
        # Analysis dictionaries
        success_patterns = {}
        growth_timelines = {}
        content_strategies = {}
        
        # Analyze each success tier
        for tier, channels in success_tiers.items():
            if not channels:
                continue
                
            # Sort by subscriber count within tier
            channels.sort(key=lambda x: x[1]['subscriber_count'], reverse=True)
            
            # Calculate average growth rate per tier
            avg_days_to_success = np.mean([data['channel_age_days'] for _, data in channels])
            avg_subscriber_growth = np.mean([data['subscriber_growth_per_day'] for _, data in channels])
            
            # Aggregate video titles to find common patterns
            all_titles = []
            for _, data in channels:
                for video in data.get('recent_videos', []):
                    all_titles.append(video['title'].lower())
            
            # Extract common words/phrases
            title_text = ' '.join(all_titles)
            title_text = re.sub(r'[^\w\s]', '', title_text)
            common_words = Counter(title_text.split()).most_common(20)
            
            # Calculate frequency of uploads (for active channels)
            upload_frequency = []
            for _, data in channels:
                videos = data.get('recent_videos', [])
                if len(videos) >= 2:
                    dates = [datetime.datetime.strptime(v['published_at'].split('T')[0], '%Y-%m-%d') 
                             for v in videos]
                    date_diffs = [(dates[i] - dates[i+1]).days for i in range(len(dates)-1)]
                    if date_diffs:  # Ensure we have differences to calculate
                        avg_days_between = sum(date_diffs) / len(date_diffs)
                        upload_frequency.append(avg_days_between)
            
            avg_upload_frequency = np.mean(upload_frequency) if upload_frequency else 0
            
            # Store patterns for this tier
            success_patterns[tier] = {
                'count': len(channels),
                'avg_days_to_current_level': avg_days_to_success,
                'avg_subscriber_growth_per_day': avg_subscriber_growth,
                'avg_upload_frequency_days': avg_upload_frequency,
                'common_title_elements': common_words,
                'top_channel_examples': [data['channel_title'] for _, data in channels[:3]]
            }
            
            # Estimate growth timeline for this tier
            if tier != 'Micro':
                # For non-micro tiers, calculate how long it took to reach current level
                growth_timelines[tier] = {
                    '1K_subscribers': np.mean([
                        data['channel_age_days'] * (1000 / max(data['subscriber_count'], 1000))
                        for _, data in channels if data['subscriber_count'] >= 1000
                    ]) if any(data['subscriber_count'] >= 1000 for _, data in channels) else None,
                    '10K_subscribers': np.mean([
                        data['channel_age_days'] * (10000 / max(data['subscriber_count'], 10000))
                        for _, data in channels if data['subscriber_count'] >= 10000
                    ]) if any(data['subscriber_count'] >= 10000 for _, data in channels) else None,
                    '100K_subscribers': np.mean([
                        data['channel_age_days'] * (100000 / max(data['subscriber_count'], 100000))
                        for _, data in channels if data['subscriber_count'] >= 100000
                    ]) if any(data['subscriber_count'] >= 100000 for _, data in channels) else None,
                    '1M_subscribers': np.mean([
                        data['channel_age_days'] * (1000000 / max(data['subscriber_count'], 1000000))
                        for _, data in channels if data['subscriber_count'] >= 1000000
                    ]) if any(data['subscriber_count'] >= 1000000 for _, data in channels) else None,
                }
        
        # Extract content strategies from successful channels (Medium+ tiers)
        successful_channels = []
        for tier in ['Mega', 'Large', 'Medium']:
            successful_channels.extend(success_tiers[tier])
        
        # Sort by subscriber growth rate
        successful_channels.sort(
            key=lambda x: x[1]['subscriber_growth_per_day'], 
            reverse=True
        )
        
        # Analyze top successful channels
        top_successful = successful_channels[:min(10, len(successful_channels))]
        for channel_id, data in top_successful:
            # Get more detailed content analysis for top channels
            try:
                recent_videos = data.get('recent_videos', [])
                
                # Calculate engagement metrics
                engagement_rates = []
                duration_seconds = []
                
                for video in recent_videos:
                    views = max(video['view_count'], 1)  # Avoid division by zero
                    engagement = (video['like_count'] + video['comment_count']) / views
                    engagement_rates.append(engagement)
                    
                    # Parse duration (PT1M30S format)
                    duration = video['duration']
                    minutes = re.search(r'(\d+)M', duration)
                    seconds = re.search(r'(\d+)S', duration)
                    total_seconds = 0
                    if minutes:
                        total_seconds += int(minutes.group(1)) * 60
                    if seconds:
                        total_seconds += int(seconds.group(1))
                    duration_seconds.append(total_seconds)
                
                content_strategies[channel_id] = {
                    'channel_title': data['channel_title'],
                    'subscriber_count': data['subscriber_count'],
                    'avg_engagement_rate': np.mean(engagement_rates) if engagement_rates else 0,
                    'avg_video_duration_seconds': np.mean(duration_seconds) if duration_seconds else 0,
                    'publishing_consistency': np.std(upload_frequency) if upload_frequency else None,
                    'rapid_growth_factor': data['subscriber_growth_per_day'] / (data['channel_age_days'] or 1)
                }
                    
            except Exception as e:
                print(f"Error analyzing content strategy for {data['channel_title']}: {e}")
                
        self.success_path_analysis = {
            'patterns_by_tier': success_patterns,
            'growth_timelines': growth_timelines,
            'content_strategies': content_strategies
        }
        
        return self.success_path_analysis

    def create_competitive_landscape(self):
        """Generate a map of the competitive landscape in this niche."""
        if not self.channel_growth_data:
            print("No channel growth data. Run analyze_channel_growth first.")
            return {}
            
        # Create a DataFrame for easier analysis
        channels_df = pd.DataFrame([
            {
                'channel_id': cid,
                'channel_title': data['channel_title'],
                'subscriber_count': data['subscriber_count'],
                'view_count': data['total_view_count'],
                'video_count': data['video_count'],
                'channel_age_days': data['channel_age_days'],
                'subs_per_day': data['subscriber_growth_per_day'],
                'subs_per_video': data['subscriber_count'] / max(data['video_count'], 1),
                'views_per_sub': data['total_view_count'] / max(data['subscriber_count'], 1),
                'success_tier': data['success_tier']
            }
            for cid, data in self.channel_growth_data.items()
        ])
        
        # Calculate quartiles for key metrics
        landscape = {
            'total_channels_analyzed': len(channels_df),
            'subscriber_distribution': {
                'min': channels_df['subscriber_count'].min(),
                'q1': channels_df['subscriber_count'].quantile(0.25),
                'median': channels_df['subscriber_count'].median(),
                'q3': channels_df['subscriber_count'].quantile(0.75),
                'max': channels_df['subscriber_count'].max(),
            },
            'age_to_success': {
                'min_days': channels_df['channel_age_days'].min(),
                'median_days': channels_df['channel_age_days'].median(),
                'max_days': channels_df['channel_age_days'].max(),
            },
            'growth_rate': {
                'min_subs_per_day': channels_df['subs_per_day'].min(),
                'median_subs_per_day': channels_df['subs_per_day'].median(),
                'max_subs_per_day': channels_df['subs_per_day'].max(),
            },
            'tier_distribution': channels_df['success_tier'].value_counts().to_dict(),
            'content_efficiency': {
                'median_subs_per_video': channels_df['subs_per_video'].median(),
                'top_quartile_subs_per_video': channels_df['subs_per_video'].quantile(0.75),
            }
        }
        
        # Identify key players (top 5 in each tier)
        key_players = {}
        for tier in channels_df['success_tier'].unique():
            tier_df = channels_df[channels_df['success_tier'] == tier]
            top_channels = tier_df.sort_values('subscriber_count', ascending=False).head(5)
            
            key_players[tier] = [
                {
                    'channel_title': row['channel_title'],
                    'subscriber_count': row['subscriber_count'],
                    'views_per_sub': row['views_per_sub'],
                    'subs_per_day': row['subs_per_day'],
                }
                for _, row in top_channels.iterrows()
            ]
        
        landscape['key_players'] = key_players
        self.competitive_landscape = landscape
        
        return landscape

    def analyze_content_evolution(self):
        """Analyze how content evolves as channels grow."""
        if not self.channel_growth_data:
            print("No channel growth data. Run analyze_channel_growth first.")
            return {}
            
        # For channels with sufficient history, analyze content changes over time
        evolution_data = {}
        
        for channel_id, data in self.channel_growth_data.items():
            # Only analyze channels with at least 10 videos in history
            recent_videos = data.get('recent_videos', [])
            if len(recent_videos) >= 10:
                
                # Sort videos by date
                videos = sorted(recent_videos, key=lambda x: x['published_at'])
                
                # Create time-ordered analysis
                time_periods = []
                period_size = max(1, len(videos) // 3)  # Split into up to 3 periods
                
                for i in range(0, len(videos), period_size):
                    period_videos = videos[i:i+period_size]
                    
                    # Calculate period metrics
                    period_data = {
                        'period': i // period_size + 1,
                        'avg_views': np.mean([v['view_count'] for v in period_videos]),
                        'avg_likes': np.mean([v['like_count'] for v in period_videos]),
                        'avg_comments': np.mean([v['comment_count'] for v in period_videos]),
                        'engagement_rate': np.mean([
                            (v['like_count'] + v['comment_count']) / max(v['view_count'], 1) 
                            for v in period_videos
                        ]),
                    }
                    
                    # Extract common title elements
                    titles = ' '.join([v['title'].lower() for v in period_videos])
                    titles = re.sub(r'[^\w\s]', '', titles)
                    period_data['common_words'] = Counter(titles.split()).most_common(5)
                    
                    time_periods.append(period_data)
                
                # Calculate content evolution metrics
                if len(time_periods) > 1:
                    view_growth = time_periods[-1]['avg_views'] / max(time_periods[0]['avg_views'], 1) - 1
                    engagement_change = time_periods[-1]['engagement_rate'] - time_periods[0]['engagement_rate']
                    
                    evolution_data[channel_id] = {
                        'channel_title': data['channel_title'],
                        'subscriber_count': data['subscriber_count'], 
                        'time_periods': time_periods,
                        'view_growth': view_growth,
                        'engagement_change': engagement_change,
                    }
        
        self.content_evolution_data = evolution_data
        return evolution_data

    def create_adaptation_strategy(self):
        """Create a customized adaptation strategy based on all analysis."""
        if not self.success_path_analysis or not self.competitive_landscape:
            print("Missing required analyses. Run analyze_success_patterns and create_competitive_landscape first.")
            return {}
            
        # Create a comprehensive adaptation strategy
        strategy = {
            'market_summary': {},
            'growth_expectations': {},
            'content_recommendations': {},
            'publishing_strategy': {},
            'audience_building': {},
            'ai_automation_opportunities': {}
        }
        
        # Market summary
        landscape = self.competitive_landscape
        strategy['market_summary'] = {
            'saturation_level': 'High' if landscape['tier_distribution'].get('Mega', 0) > 5 else 
                               'Medium' if landscape['tier_distribution'].get('Large', 0) > 10 else
                               'Low',
            'estimated_timeline_to_1k': landscape['age_to_success']['median_days'] * 0.3,
            'estimated_timeline_to_10k': landscape['age_to_success']['median_days'] * 0.7,
            'potential_subscriber_acquisition_rate': landscape['growth_rate']['median_subs_per_day'] * 1.2,
            'key_competitors': [
                player['channel_title'] for tier in landscape['key_players'].values()
                for player in tier[:3]  # Top 3 from each tier
            ][:5]  # Limit to top 5 overall
        }
        
        # Growth expectations (based on success stories)
        if self.success_path_analysis['growth_timelines']:
            best_growth = None
            for tier, timelines in self.success_path_analysis['growth_timelines'].items():
                if timelines.get('10K_subscribers'):
                    if best_growth is None or timelines['10K_subscribers'] < best_growth:
                        best_growth = timelines['10K_subscribers']
            
            if best_growth:
                strategy['growth_expectations'] = {
                    'optimistic_1k_timeline_days': best_growth * 0.1,
                    'realistic_1k_timeline_days': best_growth * 0.15,
                    'optimistic_10k_timeline_days': best_growth * 0.8,
                    'realistic_10k_timeline_days': best_growth,
                    'expected_initial_growth_rate': landscape['growth_rate']['median_subs_per_day'] * 0.5,
                    'plateau_points_to watch': [1000, 5000, 10000]  # Common plateau points
                }
        
        # Content recommendations (based on successful content and analysis)
        successful_patterns = self.success_path_analysis['patterns_by_tier'].get('Medium', {})
        common_title_elements = successful_patterns.get('common_title_elements', [])    # Safely add subreddits if available
        
        # Also incorporate our original content analysis
        if self.analysis_results:
            high_performing_words = self.analysis_results.get('high_performing_words', {})
            common_tags = self.analysis_results.get('common_tags', {})
            
            strategy['content_recommendations'] = {
                'title_formula': [word for word, _ in common_title_elements[:10] if len(word) > 2],
                'high_performing_keywords': [word for word, _ in list(high_performing_words.items())[:15]],
                'recommended_tags': [tag for tag, _ in list(common_tags.items())[:10]],
                'optimal_duration_seconds': np.mean([
                    strat.get('avg_video_duration_seconds', 30) 
                    for strat in self.success_path_analysis['content_strategies'].values()
                ]) if self.success_path_analysis.get('content_strategies') else 30,
            }
        
        # Publishing strategy
        avg_frequency = np.mean([
            patterns.get('avg_upload_frequency_days', 3) 
            for tier, patterns in self.success_path_analysis['patterns_by_tier'].items()
            if tier in ['Medium', 'Large', 'Mega']
        ]) if any(tier in self.success_path_analysis['patterns_by_tier'] for tier in ['Medium', 'Large', 'Mega']) else 3
        
        strategy['publishing_strategy'] = {
            'optimal_publishing_frequency_days': avg_frequency,
            'consistency_importance': 'High',
            'recommended_batch_size': 5,
            'youtube_algorithm_considerations': [
                "Initial 48 hours crucial for video performance",
                "Aim for >40% average view duration",
                "Optimize for click-through rate with thumbnails",
                "Engage with comments in first 2 hours"
            ]
        }
        
        # Audience building
        strategy['audience_building'] = {
            'community_engagement_tactics': [
                "Respond to every comment for first 100 videos",
                "Create community posts asking for topic suggestions",
                "Highlight top fan comments in videos",
                "Create 'series' content that builds anticipation"
            ],
            'cross_platform_strategy': [
                "Share clips on TikTok to drive YouTube traffic",
                "Create a subreddit for your channel community",
                "Use Twitter to announce new videos and engage with trend topics",
                "Consider Instagram for behind-the-scenes content"
            ]
        }
        
        # AI automation opportunities
        strategy['ai_automation_opportunities'] = {
            'content_creation': [
                "Automated tracking of trending Reddit posts across target subreddits",
                "Voice synthesis with emotional tone matching for different content types",
                "Template-based video generation with customizable elements",
                "Background music selection based on content sentiment analysis"
            ],
            'optimization': [
                "A/B testing system for titles and thumbnails",
                "Automated tagging based on content analysis",
                "Scheduled publishing at optimal times based on audience analytics",
                "Auto-generation of localized captions for international audience"
            ],
            'scaling': [
                "Multi-format output: convert successful shorts to long-form content",
                "Create topical playlists automatically grouping related shorts",
                "Develop niche-specific voice personas for different content categories",
                "Implement feedback loop where successful formats get prioritized in production"
            ]
        }
        
        return strategy

    def generate_enhanced_report(self):
        """Generate a comprehensive enhanced research report with detailed recommendations."""
        report = []
        report.append("# Enhanced YouTube Shorts Research Report: Path to Growth")
        
        # Standard report sections
        if self.analysis_results:
            report.append("\n## Content Performance Overview")
            report.append(f"Total videos analyzed: {self.analysis_results['stats']['total_videos']}")
            report.append(f"Average views: {int(self.analysis_results['stats']['avg_views']):,}")
            report.append(f"Median views: {int(self.analysis_results['stats']['median_views']):,}")
            report.append(f"Engagement ratio: {self.analysis_results['stats']['engagement_ratio']:.2%}")
            
            report.append("\n### Top Performing Content")
            report.append("Most common words in high-performing videos:")
            top_words = sorted(self.analysis_results['high_performing_words'].items(), 
                            key=lambda x: x[1], reverse=True)[:15]
            for word, count in top_words:
                if len(word) > 3:  # Filter out common short words
                    report.append(f"- {word}: {count}")
        
        # Channel growth analysis
        if hasattr(self, 'channel_growth_data') and self.channel_growth_data:
            report.append("\n## Channel Growth Analysis")
            
            # Success tier breakdown
            success_tiers = {'Micro': 0, 'Small': 0, 'Medium': 0, 'Large': 0, 'Mega': 0}
            for _, data in self.channel_growth_data.items():
                success_tiers[data['success_tier']] += 1
            
            report.append("### Success Tier Breakdown:")
            for tier, count in success_tiers.items():
                report.append(f"- {tier}: {count} channels")
            
            # Highlight success stories
            success_stories = sorted(
                [(cid, data) for cid, data in self.channel_growth_data.items() 
                if data['channel_age_days'] <= 365 and data['subscriber_count'] >= 10000],
                key=lambda x: x[1]['subscriber_count'],
                reverse=True
            )
            
            if success_stories:
                report.append("\n### Success Stories:")
                report.append("Channels that reached 10K+ subscribers within 1 year:")
                for cid, data in success_stories[:5]:
                    report.append(f"- {data['channel_title']}: {data['subscriber_count']:,} subscribers "
                                f"in {data['channel_age_days']} days")
                    # Add growth rate details
                    avg_daily_growth = data['subscriber_count'] / max(data['channel_age_days'], 1)
                    report.append(f"  * Average growth: {avg_daily_growth:.1f} subscribers/day")
                    report.append(f"  * Content frequency: {len(data.get('recent_videos', []))/10:.1f} videos/week (est.)")
        
        # Success path analysis
        if hasattr(self, 'success_path_analysis') and self.success_path_analysis:
            report.append("\n## Success Path Analysis")
            
            # Patterns by tier
            report.append("### Patterns by Success Tier:")
            for tier, patterns in self.success_path_analysis['patterns_by_tier'].items():
                report.append(f"\n#### {tier} Channels:")
                report.append(f"- Average days to current level: {patterns['avg_days_to_current_level']:.1f}")
                report.append(f"- Average subscriber growth per day: {patterns['avg_subscriber_growth_per_day']:.1f}")
                report.append(f"- Average upload frequency: {patterns['avg_upload_frequency_days']:.1f} days")
                report.append("Common title elements:")
                for word, count in patterns['common_title_elements']:
                    report.append(f"  - {word}: {count}")
                report.append("Top channel examples:")
                for channel in patterns['top_channel_examples']:
                    report.append(f"  - {channel}")
            
            # Growth timelines
            report.append("\n### Growth Timelines:")
            for tier, timelines in self.success_path_analysis['growth_timelines'].items():
                report.append(f"\n#### {tier} Channels:")
                for milestone, days in timelines.items():
                    if days is not None:
                        report.append(f"- {milestone}: {days:.1f} days")
        
        # Competitive landscape
        if hasattr(self, 'competitive_landscape') and self.competitive_landscape:
            report.append("\n## Competitive Landscape")
            landscape = self.competitive_landscape
            
            report.append("### Subscriber Distribution:")
            report.append(f"- Min: {landscape['subscriber_distribution']['min']:,}")
            report.append(f"- Q1: {landscape['subscriber_distribution']['q1']:,}")
            report.append(f"- Median: {landscape['subscriber_distribution']['median']:,}")
            report.append(f"- Q3: {landscape['subscriber_distribution']['q3']:,}")
            report.append(f"- Max: {landscape['subscriber_distribution']['max']:,}")
            
            report.append("\n### Age to Success:")
            report.append(f"- Min days: {landscape['age_to_success']['min_days']:,}")
            report.append(f"- Median days: {landscape['age_to_success']['median_days']:,}")
            report.append(f"- Max days: {landscape['age_to_success']['max_days']:,}")
            
            report.append("\n### Growth Rate:")
            report.append(f"- Min subs per day: {landscape['growth_rate']['min_subs_per_day']:.1f}")
            report.append(f"- Median subs per day: {landscape['growth_rate']['median_subs_per_day']:.1f}")
            report.append(f"- Max subs per day: {landscape['growth_rate']['max_subs_per_day']:.1f}")
            
            report.append("\n### Key Players:")
            for tier, players in landscape['key_players'].items():
                report.append(f"\n#### {tier} Channels:")
                for player in players:
                    report.append(f"- {player['channel_title']}: {player['subscriber_count']:,} subscribers, "
                                f"{player['views_per_sub']:.1f} views per sub, "
                                f"{player['subs_per_day']:.1f} subs per day")
        
        # Content evolution
        if hasattr(self, 'content_evolution_data') and self.content_evolution_data:
            report.append("\n## Content Evolution Analysis")
            
            for channel_id, data in self.content_evolution_data.items():
                report.append(f"\n### {data['channel_title']} ({data['subscriber_count']:,} subscribers)")
                for period in data['time_periods']:
                    report.append(f"- Period {period['period']}:")
                    report.append(f"  - Avg views: {period['avg_views']:.1f}")
                    report.append(f"  - Avg likes: {period['avg_likes']:.1f}")
                    report.append(f"  - Avg comments: {period['avg_comments']:.1f}")
                    report.append(f"  - Engagement rate: {period['engagement_rate']:.2%}")
                    report.append("  - Common words:")
                    for word, count in period['common_words']:
                        report.append(f"    - {word}: {count}")
                report.append(f"- View growth: {data['view_growth']:.2%}")
                report.append(f"- Engagement change: {data['engagement_change']:.2%}")
        
        # Adaptation strategy
        if hasattr(self, 'create_adaptation_strategy') and callable(getattr(self, 'create_adaptation_strategy')):
            strategy = self.create_adaptation_strategy()
            report.append("\n## Adaptation Strategy")
            
            report.append("### Market Summary:")
            for key, value in strategy['market_summary'].items():
                report.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
            
            report.append("\n### Growth Expectations:")
            for key, value in strategy['growth_expectations'].items():
                report.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
            
            report.append("\n### Content Recommendations:")
            for key, value in strategy['content_recommendations'].items():
                report.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
            
            report.append("\n### Publishing Strategy:")
            for key, value in strategy['publishing_strategy'].items():
                report.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
            
            report.append("\n### Audience Building:")
            for key, value in strategy['audience_building'].items():
                report.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
            
            report.append("\n### AI Automation Opportunities:")
            for key, value in strategy['ai_automation_opportunities'].items():
                report.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
        
        return "\n".join(report)

    def analyze_with_groq(self, prompt_template=None, model="llama3-70b-8192", analysis_type="general"):
        """Use Groq to perform advanced analysis on the collected data."""
        if not self.groq_api_key:
            logger.warning("No Groq API key provided. Set GROQ_API_KEY in .env or pass when initializing.")
            print("No Groq API key provided. Set GROQ_API_KEY in .env or pass when initializing.")
            return None
        
        # Select appropriate prompt template based on analysis type
        if not prompt_template:
            if analysis_type == "channel":
                prompt_template = """
                You are an expert YouTube channel performance analyst. I have a YouTube channel with shorts content.
                Analyze this channel data in extreme detail and provide:
                
                1. Video Success Factors: For each top performing video, identify EXACTLY what elements made it successful:
                   - Specific keywords or phrases in titles that attracted viewers
                   - Content elements that likely increased retention
                   - Emotional hooks that drove engagement
                   - Technical aspects (duration, pacing, hook timing)
                
                2. Performance Pattern Analysis:
                   - Identify hidden patterns between high vs. low performing videos
                   - Which title formulations consistently perform better?
                   - Are there specific content formats that outperform others?
                   - Day/time posting patterns that correlate with success
                
                3. Detailed Content Strategy Recommendations:
                   - Exact title structures to replicate with specific keywords
                   - Content presentation techniques to emphasize
                   - Precise hook formula based on successful videos
                   - Video length and pacing recommendations
                
                4. Audience Insights:
                   - What specific viewer needs are being met by successful videos?
                   - What psychological triggers are working most effectively?
                   - How can I better target my specific audience demographic?
                
                CHANNEL DATA:
                {data}
                """
            elif analysis_type == "niche":
                prompt_template = """
                You are an expert YouTube content strategy analyst. Analyze the following data about YouTube Shorts 
                in this niche and provide:
                
                1. Deep Pattern Analysis:
                   - Identify subtle patterns in successful videos that might not be immediately obvious
                   - Extract exact linguistic patterns in titles that correlate with higher views
                   - Map specific content themes to view counts and engagement metrics
                
                2. Competitive Gap Analysis:
                   - Identify precise underserved content niches with mathematical reasoning
                   - Calculate potential viewer capture opportunities with view estimation
                   - Map competitive density against potential viewership
                
                3. Precise Content Formula Recommendations:
                   - Specific title templates with exact keyword placement
                   - Content structure recommendations with timing (first 3 seconds, etc.)
                   - Ideal hook constructions with examples
                   - Specific editing techniques that appear to boost performance
                
                4. Trend Lifecycle Predictions:
                   - Which content themes are rising vs. declining?
                   - Mathematical growth projections for emerging niches
                   - Early indicators of algorithm shifts or preference changes
                
                5. Advanced Reddit-to-YouTube Content Strategy:
                   - Exact subreddit content adaptation techniques
                   - Specific types of Reddit posts that translate best to YouTube success
                   - Timing recommendations for content discovery to publishing
                
                DATA:
                {data}
                """
            else:
                prompt_template = """
                You are an expert YouTube content strategy analyst. Analyze the following data about YouTube Shorts 
                focusing on Reddit content and provide:
                
                1. Hidden patterns and opportunities in the data (be extremely specific):
                   - Exact keywords and phrases driving the highest engagement
                   - Precise content formats with the highest view-to-subscriber conversion
                   - Mathematical correlations between content elements and performance
                
                2. Specific recommendations for creating high-performing content:
                   - Exact title templates with specific keyword placement
                   - Precise content structure recommendations with timing
                   - Specific emotional hooks that drive the highest engagement
                
                3. Detailed predictions for future trends in this niche:
                   - Emerging content formats with mathematical growth projections
                   - Declining themes to avoid with supporting data
                   - Algorithm preference shifts based on observed patterns
                
                4. A step-by-step strategy to grow a new channel in this space:
                   - Precise content calendar with specific themes
                   - Exact title and thumbnail optimization techniques
                   - Detailed production workflow to maximize algorithm performance
                
                DATA:
                {data}
                """
        
        try:
            # Prepare data to send to Groq
            if analysis_type == "channel" and hasattr(self, 'channel_analysis'):
                # For channel analysis, focus on channel-specific data
                analysis_data = self.channel_analysis
                
                # Add additional video performance metrics
                if 'top_performing_videos' in analysis_data:
                    for video in analysis_data['top_performing_videos']:
                        # Calculate additional metrics for deeper analysis
                        if 'like_count' in video and 'view_count' in video:
                            video['like_to_view_ratio'] = video['like_count'] / max(video['view_count'], 1)
                        if 'age_days' in video and 'view_count' in video:
                            video['daily_view_velocity'] = video['view_count'] / max(video['age_days'], 1)
                        
                        # Extract title components for analysis
                        video['title_word_count'] = len(video['title'].split())
                        video['title_has_number'] = any(c.isdigit() for c in video['title'])
                        video['title_has_question'] = '?' in video['title']
                        video['title_has_emotion_words'] = False  # Could be enhanced with sentiment analysis
                
            else:
                # For general or niche analysis
                analysis_data = {
                    "content_analysis": self.analysis_results if hasattr(self, 'analysis_results') else {},
                    "trending_subreddits": self.trending_reddit_data if hasattr(self, 'trending_reddit_data') else {},
                    "untapped_niches": self.identify_untapped_niches()[:10] if hasattr(self, 'identify_untapped_niches') else [],
                }
                
                # Add advanced analysis if available
                if hasattr(self, 'success_path_analysis') and self.success_path_analysis:
                    analysis_data["success_path_analysis"] = self.success_path_analysis
                    
                if hasattr(self, 'competitive_landscape') and self.competitive_landscape:
                    analysis_data["competitive_landscape"] = self.competitive_landscape
                    
                if hasattr(self, 'content_evolution_data') and self.content_evolution_data:
                    analysis_data["content_evolution_data"] = self.content_evolution_data
                    
                if hasattr(self, 'gap_analysis'):
                    analysis_data["gap_analysis"] = self.gap_analysis
                
            # Use custom encoder to handle numpy types
            data_str = json.dumps(analysis_data, indent=2, cls=NumpyEncoder)
            
            # Prepare the request
            prompt = prompt_template.format(data=data_str)
            
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 4096
            }
            
            logger.info(f"Sending request to Groq API for {analysis_type} analysis...")
            print(f"Sending request to Groq API for {analysis_type} analysis...")
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                return analysis
            else:
                logger.error(f"Error from Groq API: {response.status_code}")
                logger.error(response.text)
                print(f"Error from Groq API: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            logger.error(traceback.format_exc())
            print(f"Error calling Groq API: {e}")
            return None

    def analyze_channel_with_groq(self, channel_analysis):
        """
        Perform advanced Groq analysis specifically for a channel's performance.
        This provides deep insights into what makes specific videos successful.
        """
        if not self.groq_api_key:
            logger.warning("No Groq API key provided. Set GROQ_API_KEY in .env or pass when initializing.")
            print("No Groq API key provided. Set GROQ_API_KEY in .env or pass when initializing.")
            return None
            
        try:
            # Store the channel analysis for Groq to analyze
            self.channel_analysis = channel_analysis
            
            # Call Groq with channel-specific analysis
            return self.analyze_with_groq(analysis_type="channel")
        except Exception as e:
            logger.error(f"Error analyzing channel with Groq: {e}")
            logger.error(traceback.format_exc())
            print(f"Error analyzing channel with Groq: {e}")
            return None

    def perform_content_gap_analysis(self):
        """Identify content types and formats that are underrepresented but have high potential."""
        if not self.shorts_data or not hasattr(self, 'trending_reddit_data'):
            print("Insufficient data. Run search_reddit_shorts and analyze_trending_subreddits first.")
            return {}
            
        # Extract content formats and themes from titles and descriptions
        formats = {
            'reaction': r'reac(t|tion)|watching|responds',
            'story': r'story|stories|happened|experience',
            'advice': r'advice|how to|tips|guide',
            'humor': r'funny|comedy|laugh|meme',
            'compilation': r'compil|best of|top|moments',
            'opinion': r'opinion|thoughts|think|hot take',
            'review': r'review|rating|stars'
        }
        
        # Extract themes and emotions
        themes = {
            'relationship': r'relationship|dating|marriage|girlfriend|boyfriend|wife|husband',
            'family': r'family|parent|child|mom|dad|brother|sister',
            'confession': r'confession|admit|secret|tifu',
            'drama': r'drama|fight|argument|conflict',
            'surprise': r'surprise|unexpected|plot twist|shocking',
            'inspirational': r'inspire|motivat|success|triumph',
            'cringe': r'cringe|awkward|embarrass',
            'justice': r'justice|revenge|karma|deserved',
            'horror': r'horror|scary|creepy|terrify|disturb'
        }
        
        # Count formats and themes
        format_counts = {format_type: 0 for format_type in formats}
        theme_counts = {theme_type: 0 for theme_type in themes}
        format_views = {format_type: [] for format_type in formats}
        theme_views = {theme_type: [] for theme_type in themes}
        
        # Process each video
        for video in self.shorts_data:
            title = video['title'].lower()
            desc = video['description'].lower()
            text = title + " " + desc
            views = video['view_count']
            
            # Check formats
            for format_type, pattern in formats.items():
                if re.search(pattern, text):
                    format_counts[format_type] += 1
                    format_views[format_type].append(views)
            
            # Check themes
            for theme_type, pattern in themes.items():
                if re.search(pattern, text):
                    theme_counts[theme_type] += 1
                    theme_views[theme_type].append(views)
        
        # Calculate average views and engagement for each format and theme
        format_analysis = {}
        for format_type in formats:
            if format_counts[format_type] > 0:
                avg_views = sum(format_views[format_type]) / format_counts[format_type]
                format_analysis[format_type] = {
                    'count': format_counts[format_type],
                    'avg_views': avg_views,
                    'saturation': format_counts[format_type] / len(self.shorts_data)
                }
        
        theme_analysis = {}
        for theme_type in themes:
            if theme_counts[theme_type] > 0:
                avg_views = sum(theme_views[theme_type]) / theme_counts[theme_type]
                theme_analysis[theme_type] = {
                    'count': theme_counts[theme_type],
                    'avg_views': avg_views,
                    'saturation': theme_counts[theme_type] / len(self.shorts_data)
                }
        
        # Identify high-performing but under-saturated combinations
        combinations = []
        
        for format_type, format_stats in format_analysis.items():
            for theme_type, theme_stats in theme_analysis.items():
                # Calculate potential for this combination
                potential = (format_stats['avg_views'] + theme_stats['avg_views']) / 2
                saturation = (format_stats['saturation'] + theme_stats['saturation']) / 2
                
                # Assign a potential score (higher views with lower saturation = better)
                potential_score = potential * (1 - saturation)
                
                combinations.append({
                    'format': format_type,
                    'theme': theme_type,
                    'potential_score': potential_score,
                    'avg_views': potential,
                    'saturation': saturation
                })
        
        # Sort combinations by potential score
        combinations.sort(key=lambda x: x['potential_score'], reverse=True)
        
        # Identify top subreddits for each promising combination
        top_combinations = combinations[:10]
        for combo in top_combinations:
            # Find relevant subreddits for this theme
            related_subs = []
            for subreddit, stats in self.trending_reddit_data.items():
                # Check if subreddit name or description matches the theme
                if re.search(themes[combo['theme']], subreddit.lower()):
                    related_subs.append({
                        'subreddit': subreddit,
                        'avg_views': stats['avg_views'],
                        'mention_count': stats['mention_count']
                    })
            
            combo['recommended_subreddits'] = sorted(related_subs, 
                                                    key=lambda x: x['avg_views'], 
                                                    reverse=True)[:3]
        
        gap_analysis = {
            'format_analysis': format_analysis,
            'theme_analysis': theme_analysis,
            'top_combinations': top_combinations
        }
        
        return gap_analysis
    
    def save_enhanced_research(self, filename='enhanced_youtube_research.json'):
        """Save all research data including enhanced analyses to file."""
        research_data = {
            'shorts_data': self.shorts_data, 
            'trending_reddit_data': self.trending_reddit_data,
            'analysis_results': self.analysis_results,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add enhanced analyses if available
        if hasattr(self, 'channel_growth_data'):
            research_data['channel_growth_data'] = self.channel_growth_data
            
        if hasattr(self, 'success_path_analysis'):
            research_data['success_path_analysis'] = self.success_path_analysis
        
        if hasattr(self, 'competitive_landscape'):
            research_data['competitive_landscape'] = self.competitive_landscape
            
        if hasattr(self, 'content_evolution_data'):
            research_data['content_evolution_data'] = self.content_evolution_data
            
        # Use custom encoder to handle numpy types
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(research_data, f, indent=2, cls=NumpyEncoder)
            
        print(f"Enhanced research data saved to {filename}")
    
    def analyze_your_channel(self, channel_id=None, channel_username=None):
        """
        Analyze your own YouTube channel's performance and provide specific recommendations.
        Provide either channel_id or channel_username.
        """
        if not channel_id and not channel_username:
            print("Error: You must provide either a channel ID or username.")
            return None
            
        # Get channel ID if username is provided
        if not channel_id and channel_username:
            try:
                channels_response = self.youtube.channels().list(
                    part="id",
                    forUsername=channel_username
                ).execute()
                if not channels_response['items']:
                    print(f"No channel found with username: {channel_username}")
                    return None
                    
                channel_id = channels_response['items'][0]['id']
            except Exception as e:
                print(f"Error finding channel by username: {e}")
                return None
            
        # Get channel details
        try:
            channel_response = self.youtube.channels().list(
                part="snippet,statistics,contentDetails,brandingSettings",
                id=channel_id
            ).execute()
            if not channel_response['items']:
                print(f"No channel found with ID: {channel_id}")
                return None
                
            channel_info = channel_response['items'][0]
            
            # Extract basic channel data
            channel_data = {
                'id': channel_id,
                'title': channel_info['snippet']['title'],
                'description': channel_info['snippet']['description'],
                'creation_date': channel_info['snippet']['publishedAt'],
                'thumbnail_url': channel_info['snippet']['thumbnails']['high']['url'],
                'subscriber_count': int(channel_info['statistics'].get('subscriberCount', 0)),
                'video_count': int(channel_info['statistics'].get('videoCount', 0)),
                'view_count': int(channel_info['statistics'].get('viewCount', 0)),
                'keywords': channel_info.get('brandingSettings', {}).get('channel', {}).get('keywords', '')
            }
            
            # Calculate channel age in days
            creation_date = datetime.datetime.strptime(channel_data['creation_date'].split('T')[0], '%Y-%m-%d')
            today = datetime.datetime.now()
            channel_data['age_days'] = (today - creation_date).days
            channel_data['avg_daily_views'] = channel_data['view_count'] / max(channel_data['age_days'], 1)
            channel_data['avg_daily_subscribers'] = channel_data['subscriber_count'] / max(channel_data['age_days'], 1)
            
            # Get upload playlist ID
            uploads_playlist_id = channel_info['contentDetails']['relatedPlaylists']['uploads']
            
            # Get videos
            videos_data = []
            next_page_token = None
            
            # Fetch up to 50 most recent videos
            while len(videos_data) < 50:
                playlist_response = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()
                
                # Extract video IDs
                video_ids = [item['contentDetails']['videoId'] for item in playlist_response['items']]
                
                # Get detailed video stats
                if video_ids:
                    videos_response = self.youtube.videos().list(
                        part="statistics,snippet,contentDetails,topicDetails",
                        id=','.join(video_ids)
                    ).execute()
                    
                    for video in videos_response['items']:
                        # Determine if it's a short
                        is_short = False
                        title = video['snippet']['title'].lower()
                        description = video['snippet']['description'].lower()
                        tags = video['snippet'].get('tags', [])
                        
                        # Check for vertical video aspect ratio or "#shorts" markers
                        if ('shorts' in title or '#shorts' in title or 
                            'shorts' in description or '#shorts' in description or 
                            'shorts' in [tag.lower() for tag in tags]):
                            is_short = True
                        
                        # Extract video data
                        video_data = {
                            'id': video['id'],
                            'title': video['snippet']['title'],
                            'description': video['snippet']['description'],
                            'publish_date': video['snippet']['publishedAt'],
                            'thumbnail_url': video['snippet']['thumbnails']['high']['url'] if 'high' in video['snippet']['thumbnails'] else None,
                            'tags': video['snippet'].get('tags', []),
                            'view_count': int(video['statistics'].get('viewCount', 0)),
                            'like_count': int(video['statistics'].get('likeCount', 0)),
                            'comment_count': int(video['statistics'].get('commentCount', 0)),
                            'duration': video['contentDetails']['duration'],
                            'is_short': is_short,
                            'topics': video.get('topicDetails', {}).get('topicCategories', [])
                        }
                        
                        # Parse duration (PT1M30S format)
                        duration = video_data['duration']
                        minutes = re.search(r'(\d+)M', duration)
                        seconds = re.search(r'(\d+)S', duration)
                        hours = re.search(r'(\d+)H', duration)
                        
                        total_seconds = 0
                        if hours:
                            total_seconds += int(hours.group(1)) * 3600
                        if minutes:
                            total_seconds += int(minutes.group(1)) * 60
                        if seconds:
                            total_seconds += int(seconds.group(1))
                            
                        video_data['duration_seconds'] = total_seconds
                        
                        # Calculate video age and performance metrics
                        publish_date = datetime.datetime.strptime(video_data['publish_date'].split('T')[0], '%Y-%m-%d')
                        video_age_days = max((today - publish_date).days, 1)  # Avoid division by zero
                        video_data['age_days'] = video_age_days
                        video_data['views_per_day'] = video_data['view_count'] / video_age_days
                        video_data['engagement_rate'] = (video_data['like_count'] + video_data['comment_count']) / max(video_data['view_count'], 1)
                        
                        videos_data.append(video_data)
                
                # Check for more pages
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token or len(videos_data) >= 50:
                    break
            
            # Separate shorts from regular videos
            shorts_data = [v for v in videos_data if v['is_short']]
            regular_videos = [v for v in videos_data if not v['is_short']]
            
            # Analyze shorts performance
            if shorts_data:
                shorts_analysis = {
                    'count': len(shorts_data),
                    'avg_views': np.mean([v['view_count'] for v in shorts_data]),
                    'median_views': np.median([v['view_count'] for v in shorts_data]),
                    'avg_duration': np.mean([v['duration_seconds'] for v in shorts_data]),
                    'avg_engagement': np.mean([v['engagement_rate'] for v in shorts_data]),
                    'top_performer': sorted(shorts_data, key=lambda x: x['view_count'], reverse=True)[0]['title'],
                    'top_performer_views': sorted(shorts_data, key=lambda x: x['view_count'], reverse=True)[0]['view_count']
                }
                
                # Analyze title patterns in successful shorts
                if len(shorts_data) >= 3:
                    # Define successful shorts as those with above-average views
                    avg_views = shorts_analysis['avg_views']
                    successful_shorts = [v for v in shorts_data if v['view_count'] > avg_views]
                    
                    # Extract common words
                    if successful_shorts:
                        all_titles = ' '.join([s['title'].lower() for s in successful_shorts])
                        all_titles = re.sub(r'[^\w\s]', '', all_titles)
                        word_counts = Counter(all_titles.split()).most_common(15)
                        shorts_analysis['common_title_elements'] = word_counts
            else:
                shorts_analysis = {'count': 0, 'message': 'No shorts found on this channel'}
            
            # Calculate posting frequency
            if len(videos_data) >= 2:
                publish_dates = [datetime.datetime.strptime(v['publish_date'].split('T')[0], '%Y-%m-%d') 
                               for v in videos_data]
                publish_dates.sort()
                
                # Calculate days between posts
                days_between = [(publish_dates[i] - publish_dates[i+1]).days 
                              for i in range(len(publish_dates)-1)]
                
                posting_frequency = {
                    'avg_days_between_posts': np.mean(days_between) if days_between else 0,
                    'consistency_score': 1 - (np.std(days_between) / max(np.mean(days_between), 1)) if days_between else 0,
                    'recent_frequency': np.mean(days_between[:5]) if len(days_between) >= 5 else np.mean(days_between) if days_between else 0
                }
            else:
                posting_frequency = {'message': 'Not enough videos to calculate posting frequency'}
            
            # Compare to niche benchmarks (if we have analyzed other channels)
            benchmark_comparison = {}
            if hasattr(self, 'competitive_landscape') and self.competitive_landscape:
                landscape = self.competitive_landscape
                
                # Determine where this channel fits in the competitive landscape
                benchmark_comparison = {
                    'subscriber_percentile': self._get_percentile(channel_data['subscriber_count'], 
                                                                 landscape['subscriber_distribution']),
                    'growth_rate_percentile': self._get_percentile(channel_data['avg_daily_subscribers'], 
                                                                 landscape['growth_rate']),
                }
                
                # Determine success tier
                if channel_data['subscriber_count'] >= 1000000:
                    benchmark_comparison['success_tier'] = "Mega"
                elif channel_data['subscriber_count'] >= 100000:
                    benchmark_comparison['success_tier'] = "Large"
                elif channel_data['subscriber_count'] >= 10000:
                    benchmark_comparison['success_tier'] = "Medium"
                elif channel_data['subscriber_count'] >= 1000:
                    benchmark_comparison['success_tier'] = "Small"
                else:
                    benchmark_comparison['success_tier'] = "Micro"
                        
            # Generate improvement recommendations
            recommendations = self._generate_channel_recommendations(
                channel_data, shorts_analysis, posting_frequency, benchmark_comparison
            )
            
            # Return complete analysis
            channel_analysis = {
                'channel_data': channel_data,
                'shorts_analysis': shorts_analysis,
                'posting_frequency': posting_frequency,
                'benchmark_comparison': benchmark_comparison,
                'recommendations': recommendations,
                'top_performing_videos': sorted(videos_data, key=lambda x: x['view_count'], reverse=True)[:5]
            }
            
            # Store the channel analysis for potential Groq analysis later
            self.channel_analysis = channel_analysis
            
            return channel_analysis
        
        except Exception as e:
            print(f"Error analyzing channel: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_percentile(self, value, distribution):
        """Calculate the percentile of a value within a distribution."""
        if not distribution:
            return None
        
        if value <= distribution.get('min', 0):
            return 0
        if value >= distribution.get('max', float('inf')):
            return 100
        
        # Check where it falls in quartiles
        if value <= distribution.get('q1', float('inf')):
            # Between min and Q1
            range_min = distribution.get('min', 0)
            range_max = distribution.get('q1', float('inf'))
            position = (value - range_min) / (range_max - range_min) if range_max > range_min else 0
            return position * 25  # 0-25%
        elif value <= distribution.get('median', float('inf')):
            # Between Q1 and median
            range_min = distribution.get('q1', 0)
            range_max = distribution.get('median', float('inf'))
            position = (value - range_min) / (range_max - range_min) if range_max > range_min else 0
            return 25 + position * 25  # 25-50%
        elif value <= distribution.get('q3', float('inf')):
            # Between median and Q3
            range_min = distribution.get('median', 0)
            range_max = distribution.get('q3', float('inf'))
            position = (value - range_min) / (range_max - range_min) if range_max > range_min else 0
            return 50 + position * 25  # 50-75%
        else:
            # Between Q3 and max
            range_min = distribution.get('q3', 0)
            range_max = distribution.get('max', float('inf'))
            position = (value - range_min) / (range_max - range_min) if range_max > range_min else 0
            return 75 + position * 25  # 75-100%
    
    def _generate_channel_recommendations(self, channel_data, shorts_analysis, posting_frequency, benchmark_comparison):
        """Generate specific recommendations to improve channel performance."""
        recommendations = []
        
        # Title optimization recommendations
        if shorts_analysis.get('count', 0) > 0:
            if 'common_title_elements' in shorts_analysis:
                high_performing_title_elements = [word for word, count in shorts_analysis['common_title_elements'] 
                                                if len(word) > 3][:5]
                if high_performing_title_elements:
                    recommendations.append(
                        f"Title Optimization: Continue using these keywords that perform well in your shorts: "
                        f"{', '.join(high_performing_title_elements)}"
                    )
            
            # Compare with niche keyword trends
            if hasattr(self, 'analysis_results') and self.analysis_results:
                niche_keywords = [k for k, v in self.analysis_results.get('high_performing_words', {}).items() 
                                 if len(k) > 3][:10]
                recommendations.append(
                    f"Trending Keywords: Consider incorporating these trending keywords in your niche: "
                    f"{', '.join(niche_keywords)}"
                )
        
        # Posting frequency recommendations
        if isinstance(posting_frequency, dict) and 'avg_days_between_posts' in posting_frequency:
            avg_days = posting_frequency['avg_days_between_posts']
            consistency = posting_frequency.get('consistency_score', 0)
            
            if avg_days > 7:
                recommendations.append(
                    f"Posting Frequency: Your average posting interval is {avg_days:.1f} days. "
                    f"Consider posting more frequently (at least weekly) to maintain algorithm favor."
                )
            if consistency < 0.5:
                recommendations.append(
                    "Consistency: Your posting schedule appears inconsistent. Create a regular posting schedule "
                    "to help the algorithm and audience know when to expect new content."
                )
        
        # Benchmark-based recommendations
        if benchmark_comparison:
            sub_percentile = benchmark_comparison.get('subscriber_percentile')
            growth_percentile = benchmark_comparison.get('growth_rate_percentile')
            
            if sub_percentile is not None and sub_percentile < 50:
                recommendations.append(
                    f"Channel Size: Your channel is in the bottom {sub_percentile:.0f}% of the competitive landscape. "
                    f"Focus on subscriber growth by including clear calls-to-action in your videos."
                )
            if growth_percentile is not None and growth_percentile < 50:
                recommendations.append(
                    f"Growth Rate: Your subscriber growth rate is in the bottom {growth_percentile:.0f}% of competitors. "
                    f"Consider collaborations and cross-promotion to accelerate growth."
                )
        
        # Content gaps analysis
        if hasattr(self, 'perform_content_gap_analysis'):
            recommendations.append(
                "Content Opportunity: Based on our analysis, consider exploring untapped content combinations like "
                "reaction videos on relationship topics or story-based videos about surprising experiences."
            )
        
        # Untapped subreddit recommendations
        if hasattr(self, 'identify_untapped_niches') and callable(getattr(self, 'identify_untapped_niches')):
            untapped = self.identify_untapped_niches()[:3]
            if untapped:
                sub_recommendations = ", ".join([f"r/{niche['subreddit']}" for niche in untapped])
                recommendations.append(
                    f"Content Sources: Explore these underutilized subreddits for fresh content: {sub_recommendations}"
                )
        
        # Short duration optimization
        if shorts_analysis.get('count', 0) > 0 and 'avg_duration' in shorts_analysis:
            avg_duration = shorts_analysis['avg_duration']
            if avg_duration > 45:
                recommendations.append(
                    f"Shorts Duration: Your average short is {avg_duration:.1f} seconds. Consider creating more "
                    f"concise shorts (25-40 seconds) to optimize for completion rate."
                )
            elif avg_duration < 20:
                recommendations.append(
                    f"Shorts Duration: Your average short is {avg_duration:.1f} seconds. While short content can "
                    f"work well for retention, consider slightly longer shorts (25-40 seconds) to include more value."
                )
        
        # Add general best practices
        recommendations.append(
            "Engagement Strategy: Respond to comments within the first hour of posting to boost algorithm favor. "
            "Ask engaging questions in your content to encourage viewer interaction."
        )
        
        return recommendations

    def generate_channel_report(self, channel_analysis):
        """Generate a human-readable report from channel analysis."""
        if not channel_analysis:
            return "No channel analysis data available."
            
        channel_data = channel_analysis.get('channel_data', {})
        shorts_analysis = channel_analysis.get('shorts_analysis', {})
        recommendations = channel_analysis.get('recommendations', [])
        top_videos = channel_analysis.get('top_performing_videos', [])
        
        report = []
        report.append(f"# Channel Analysis Report: {channel_data.get('title', 'Unknown')}")
        report.append(f"\n## Channel Overview")
        report.append(f"- Subscribers: {channel_data.get('subscriber_count', 0):,}")
        report.append(f"- Total Views: {channel_data.get('view_count', 0):,}")
        report.append(f"- Videos: {channel_data.get('video_count', 0)}")
        report.append(f"- Channel Age: {channel_data.get('age_days', 0):,} days")
        report.append(f"- Avg. Daily Views: {channel_data.get('avg_daily_views', 0):.1f}")
        report.append(f"- Avg. Daily New Subscribers: {channel_data.get('avg_daily_subscribers', 0):.1f}")
        
        if shorts_analysis and shorts_analysis.get('count', 0) > 0:
            report.append(f"\n## Shorts Performance")
            report.append(f"- Total Shorts: {shorts_analysis.get('count', 0)}")
            report.append(f"- Average Views: {shorts_analysis.get('avg_views', 0):.1f}")
            report.append(f"- Median Views: {shorts_analysis.get('median_views', 0)::.1f}")
            report.append(f"- Average Duration: {shorts_analysis.get('avg_duration', 0):.1f} seconds")
            report.append(f"- Average Engagement Rate: {shorts_analysis.get('avg_engagement', 0):.2%}")
            report.append(f"- Top Performing Short: {shorts_analysis.get('top_performer', 'N/A')}")
            report.append(f"  with {shorts_analysis.get('top_performer_views', 0):,} views")
            
            if 'common_title_elements' in shorts_analysis:
                report.append("\n### Successful Title Elements:")
                for word, count in shorts_analysis['common_title_elements'][:10]:
                    if len(word) > 3:  # Filter out common short words
                        report.append(f"- {word}: {count}")
        
        if top_videos:
            report.append(f"\n## Top Performing Videos")
            for idx, video in enumerate(top_videos, 1):
                report.append(f"{idx}. {video['title']}")
                report.append(f"   Views: {video['view_count']:,} | Likes: {video['like_count']:,} | Comments: {video['comment_count']:,}")
                report.append(f"   Engagement Rate: {video['engagement_rate']:.2%} | Duration: {video['duration_seconds']} seconds")
                report.append(f"   Published: {video['publish_date'].split('T')[0]} ({video['age_days']} days ago)")
                report.append("")
        
        if recommendations:
            report.append(f"\n## Recommendations")
            for idx, rec in enumerate(recommendations, 1):
                report.append(f"{idx}. {rec}")
        
        return "\n".join(report)


if __name__ == "__main__":
    print("This module contains the YouTubeRedditResearcher class.")
    print("Use run_research.py to execute research with the configured settings.")