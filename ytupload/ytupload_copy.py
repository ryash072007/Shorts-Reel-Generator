import os
import time
from datetime import datetime, timezone, timedelta
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class YoutubeUploader:
    def __init__(self, credentials_folders):
        self.credentials_folders = credentials_folders
        self.scopes = [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.force-ssl",
            "https://www.googleapis.com/auth/youtube",
        ]
        self.youtube = None

    def scoped(self):
        client_secrets_path = os.path.join(
            self.credentials_folders, "client_secrets.json"
        )
        self.youtube = self._get_youtube_client(client_secrets_path)

    def _get_youtube_client(self, client_secrets_path):
        """Get authenticated YouTube client with token persistence"""
        creds = None
        token_path = client_secrets_path.replace(
            "client_secrets.json", "youtube_token.json"
        )

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.scopes)

        if not creds or not creds.valid:
            # Try different ports if default port is blocked
            ports = [8080, 8090, 8000, 8888]
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_path, self.scopes, redirect_uri="http://localhost:8080"
            )

            for port in ports:
                try:
                    print(
                        f"Attempting to start authentication server on port {port}..."
                    )
                    creds = flow.run_local_server(
                        port=port,
                        prompt="consent",
                        access_type="offline",
                        timeout_seconds=120,
                    )
                    print(f"Successfully authenticated using port {port}")
                    break
                except Exception as e:
                    print(f"Failed to use port {port}: {str(e)}")
                    if port == ports[-1]:
                        raise Exception(
                            "Failed to authenticate: Could not bind to any available ports"
                        )
                    continue

            # Save the credentials for future use
            with open(token_path, "w") as token:
                token.write(creds.to_json())

        return build("youtube", "v3", credentials=creds)

    def upload_videos(self, videos_data, retry_attempts=3, retry_delay=3600):
        """Upload multiple videos with credential rotation and retry logic"""
        for video_path, metadata in videos_data.items():
            video_uploaded = False

            for folder in self.credentials_folders:
                if video_uploaded:
                    break

                for attempt in range(retry_attempts):
                    try:
                        # Prepare video upload
                        request_body = {
                            "snippet": {
                                "title": metadata["title"],
                                "description": metadata["description"]
                                + " #shorts #youtubeshorts #viral #trending #funny #comedy",
                                "categoryId": "22",
                            },
                            "status": {
                                "privacyStatus": (
                                    "private"
                                    if "schedule_time" in metadata
                                    else "public"
                                ),
                                "selfDeclaredMadeForKids": False,
                            },
                        }

                        # Add scheduling if specified
                        if "schedule_time" in metadata:
                            schedule_time = datetime.fromisoformat(
                                metadata["schedule_time"]
                            )
                            # Convert to UTC timestamp
                            utc_timestamp = schedule_time.astimezone(timezone.utc)
                            request_body["status"][
                                "publishAt"
                            ] = utc_timestamp.isoformat()
                            print(f"Video will be published at: {utc_timestamp}")

                        media = MediaFileUpload(video_path, resumable=True)
                        insert_request = self.youtube.videos().insert(
                            part=",".join(request_body.keys()),
                            body=request_body,
                            media_body=media,
                        )

                        response = insert_request.execute()
                        video_id = response["id"]

                        # Handle pinned comment
                        if metadata.get("pinned_comment"):
                            self._add_pinned_comment(
                                self.youtube, video_id, metadata["pinned_comment"]
                            )

                        print(f"Successfully uploaded video: {metadata['title']}")
                        print(f"Video ID: {video_id}")
                        video_uploaded = True
                        break

                    except Exception as e:
                        if "quotaExceeded" in str(e) or "uploadLimitExceeded" in str(e):
                            print(
                                f"Upload limit reached. Retrying in {retry_delay} seconds..."
                            )
                            time.sleep(retry_delay)
                        else:
                            print(f"Error uploading {video_path}: {str(e)}")
                            break

                print(f"Switching to next credentials set in: {folder}")

            if not video_uploaded:
                print(f"Failed to upload video {video_path} after all attempts")
            else:
                return True

    def _add_pinned_comment(self, youtube, video_id, comment_text):
        """Helper method to add and pin a comment with retry mechanism"""
        max_retries = 3
        retry_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                # Wait a bit before adding comment to ensure video is fully processed
                time.sleep(30)  # Wait 30 seconds after video upload

                comment_response = (
                    youtube.commentThreads()
                    .insert(
                        part="snippet",
                        body={
                            "snippet": {
                                "videoId": video_id,
                                "topLevelComment": {
                                    "snippet": {"textOriginal": comment_text}
                                },
                            }
                        },
                    )
                    .execute()
                )

                # Wait briefly before trying to pin
                time.sleep(5)

                # Try to pin the comment
                try:
                    youtube.comments().setModerationStatus(
                        id=comment_response["snippet"]["topLevelComment"]["id"],
                        moderationStatus="published",
                    ).execute()
                except Exception as pin_error:
                    print(
                        f"Warning: Could not pin comment, but comment was posted: {str(pin_error)}"
                    )

                print("Successfully added comment")
                return True

            except Exception as e:
                print(f"Attempt {attempt + 1} failed to add comment: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Failed to add comment after all attempts")
                    return False


uploader = None


def init_file():
    global uploader
    # Define your credentials folders
    credentials_folders = "AiStoryMaker/cred_set_1"

    # Initialize the uploader
    uploader = YoutubeUploader(credentials_folders)

    uploader.scoped()


def create_schedule_time(year, month, day, hour=0, minute=0, tz_offset=0):
    """
    Create a properly formatted schedule time string
    :param year: Year (e.g. 2024)
    :param month: Month (1-12)
    :param day: Day (1-31)
    :param hour: Hour in 24h format (0-23), defaults to 0
    :param minute: Minute (0-59), defaults to 0
    :param tz_offset: Timezone offset in hours (e.g. -5 for EST), defaults to 0 (UTC)
    :return: Formatted datetime string
    """
    # Create datetime object
    dt = datetime(year, month, day, hour, minute)

    # Add timezone offset
    tz_delta = timezone(timedelta(hours=tz_offset))
    dt = dt.replace(tzinfo=tz_delta)

    # Convert to ISO format
    return dt.isoformat()


def main(data, schedule_time=None):
    """
    Upload videos with optional scheduling
    :param data: Dictionary containing video data
    :param schedule_time: Optional ISO format datetime string (e.g. "2024-01-01T15:00:00+00:00")
    """
    global uploader

    # Add schedule time to all videos if provided
    if schedule_time:
        for video_data in data.values():
            video_data["schedule_time"] = schedule_time

    # Upload the videos with default retry settings
    uploader.upload_videos(data)


if __name__ == "__main__":
    # Example usage with immediate upload

    # Example usage with scheduled upload using helper function
    # main(
    #     {
    #         "AiStoryMaker/output/output.mp4": {
    #             "title": "Scheduled Video",
    #             "description": "Example description",
    #             "pinned_comment": "Example comment",
    #         }
    #     },
    #     schedule_time=schedule,
    # )

    init_file()

    schedule = create_schedule_time(
        year=2025, month=2, day=15, hour=12, minute=0, tz_offset=4  # 3 PM  # UTC
    )

    main(
        # {
        #     "meme2video/output/meme_video.mp4": {
        #         "title": "When Adulting Hits You with Reality",
        #         "description": "Laughing at my problems? Nope! I'm just laughing at my empty bank account Anyone else with me? Share your funny finance struggles in the comments below! #AdultingStruggles #FinancialWoes #MemeLife #LaughterIsTheBestMedicine #FinancialLiteracy #MoneyProblems #AdultingHacks",
        #         # "pinned_comment": "Same here! Who else is guilty of laughing away their financial stress? ",
        #     }
        # },
        {
            "meme2video/output/meme_video.mp4": {
                "title": "Seagull's Perfect Heist: From Snacks to Umbrella",
                "description": "Watch as a clever seagull takes over my beach day, starting with a snack attack from miles away, inhaling my entire picnic and moving on to my beach umbrella. Who knew birds could be such masters of timing and theft? 🦩☀️ #BeachLife #SeagullTheft #PicnicFail #BeachHumor #BirdWatchers #FunnyAnimals #VacatiionFails #SummerLaughs",
                "pinned_comment": "Haha, who else has had a seagull steal their snacks on the beach? It turns out they're pretty creative thieves! 😉 Have you had any fun beach day mishaps like this? Share your stories below! #BeachComedy #SeagullWarriors",
            }
        }
        # schedule_time=schedule,
    )
