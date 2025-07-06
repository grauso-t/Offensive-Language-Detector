import os
import yt_dlp
from typing import Optional, Dict, Any
from urllib.parse import urlparse

# Download folder path
DOWNLOAD_DIR = os.path.abspath("video_downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def is_tiktok_url(url):
    """Check if the URL belongs to TikTok"""
    try:
        parsed = urlparse(url)
        tiktok_domains = [
            'tiktok.com',
            'www.tiktok.com',
            'm.tiktok.com',
            'vm.tiktok.com'
        ]
        return parsed.netloc.lower() in tiktok_domains
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """Clean filename by removing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def get_video_info(url: str) -> Optional[Dict[str, Any]]:
    """Get video information without downloading it."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'uploader': info.get('uploader', 'Unknown'),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'description': info.get('description', '')
            }
    except Exception as e:
        print(f"Unable to get video info: {e}")
        return None

def download_tiktok_video(url: str, preferred_browser: str = 'firefox', 
                         quality: str = 'best', custom_filename: str = None) -> bool:
    """
    Download a TikTok video with fallback on different browsers and options.
    
    Args:
        url: TikTok video URL
        preferred_browser: Preferred browser for cookies
        quality: Video quality ('best', 'worst', 'mp4', etc.)
        custom_filename: Custom filename (optional)
    
    Returns:
        bool: True if download succeeded, False otherwise
    """
    
    # List of browsers to try in order
    browsers_to_try = [
        preferred_browser,
        'chrome',
        'firefox',
        'safari',
        'edge',
        'opera'
    ]
    
    # Remove duplicates while maintaining order
    browsers_to_try = list(dict.fromkeys(browsers_to_try))
    
    # Get video information
    print(f"Analyzing video...")
    video_info = get_video_info(url)
    if video_info:
        print(f"Title: {video_info['title']}")
        print(f"Author: {video_info['uploader']}")
        print(f"Duration: {video_info['duration']}s")
        print(f"Views: {video_info['view_count']}")
    
    # Configure filename
    if custom_filename:
        filename_template = os.path.join(DOWNLOAD_DIR, f"{sanitize_filename(custom_filename)}.%(ext)s")
    else:
        filename_template = os.path.join(DOWNLOAD_DIR, "%(uploader)s_%(title)s_%(id)s.%(ext)s")
    
    # Base configuration
    base_opts = {
        'outtmpl': filename_template,
        'format': quality,
        'quiet': False,
        'no_warnings': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['it', 'en'],
        'ignoreerrors': True,
        'retries': 3,
        'fragment_retries': 3,
        'extractaudio': False,
        'audioformat': 'mp3',
        'embed_metadata': True,
    }
    
    # Attempt 1: With browser cookies
    for browser in browsers_to_try:
        print(f"\nAttempting with {browser} cookies...")
        
        ydl_opts = base_opts.copy()
        try:
            ydl_opts['cookiesfrombrowser'] = (browser,)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                print(f"Download completed with {browser}")
                return True
                
        except Exception as e:
            print(f"Error with {browser}: {e}")
            continue
    
    # Attempt 2: Without cookies
    print(f"\nAttempting without browser cookies...")
    try:
        ydl_opts = base_opts.copy()
        if 'cookiesfrombrowser' in ydl_opts:
            del ydl_opts['cookiesfrombrowser']
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            print("Download completed without cookies")
            return True
            
    except Exception as e:
        print(f"Error without cookies: {e}")
    
    # Attempt 3: With custom User-Agent
    print(f"\nAttempting with custom User-Agent...")
    try:
        ydl_opts = base_opts.copy()
        ydl_opts['http_headers'] = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            print("Download completed with custom User-Agent")
            return True
            
    except Exception as e:
        print(f"Error with User-Agent: {e}")
    
    # Attempt 4: Alternative format
    print(f"\nAttempting with alternative format...")
    try:
        ydl_opts = base_opts.copy()
        ydl_opts['format'] = 'worst'  # Try with lower quality
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            print("Download completed with alternative format")
            return True
            
    except Exception as e:
        print(f"Error with alternative format: {e}")
    
    print("All download attempts failed")
    return False