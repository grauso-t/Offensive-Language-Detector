import yt_dlp
import os
import audio_utils

def is_tiktok_url(url):
    """
    Check if the URL is a TikTok URL
    """
    if 'tiktok.com' in url or 'vm.tiktok.com' in url:
        return True
    else:
        return False

def process_tiktok(url):
    """
    Process and download only audio from a TikTok URL
    """
    browsers_to_try = [
        'firefox',
        'chrome',
        'safari',
        'edge',
        'opera'
    ]
   
    # Create temp directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
   
    # Get video information
    print(f"Analyzing video...")

    # Base configuration for audio extraction
    base_opts = {
        'outtmpl': "temp/%(id)s.%(ext)s",
        'format': "bestaudio/best",
        'quiet': False,
        'no_warnings': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'ignoreerrors': True,
        'retries': 3,
        'fragment_retries': 3,
        'extractaudio': True,
        'audioformat': 'mp3',
        'audioquality': '192K',
        'embed_metadata': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    # Download audio using cookies from each browser
    for browser in browsers_to_try:
        print(f"\nAttempting with {browser} cookies...")
       
        ydl_opts = base_opts.copy()

        try:
            ydl_opts['cookiesfrombrowser'] = (browser,)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                user = info.get('uploader', 'Unknown') or info.get('uploader_id', 'Unknown')
                date = info.get('upload_date', 'Unknown')
                file_path = os.path.join("temp", f"{info['id']}.mp3")
                content = audio_utils.get_audio_content(file_path)
                os.remove(file_path)

                print(f"Audio download completed")
                return {
                    'user': user,
                    'date': date,
                    'content': content,
                    'url': url,
                }
               
        except Exception as e:
            print(f"Error with {browser}: {e}")
            return None
   
    print("All download attempts failed")
    return None