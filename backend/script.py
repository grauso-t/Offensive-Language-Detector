import tiktok
import threads
import ai_utils
import json

def process_url(url):
    """
    Process the provided URL to extract content from TikTok or Threads.
    """
    if tiktok.is_tiktok_url(url):
        print("Processing TikTok URL...")
        return tiktok.process_tiktok(url)

    if threads.is_threads_url(url):
        print("Processing Threads URL...")
        return threads.process_threads(url)

    else:
        return 'Unsupported URL. We only support TikTok and Threads.'

url = input("Enter a TikTok or Threads URL: ").strip()
text = process_url(url)

if text is None:
    print("No content found in the provided URL.")
    exit()
    
print("Extracted content:")
print(f"  User   : {text['user']}")
print(f"  Date   : {text['date']}")
print(f"  Content: {text['content']}")
print(f"  URL    : {text['url']}")

if text["content"] is not None:
    result = ai_utils.is_offensive_logistic_regression(text["content"])
    print(f"Detected: {result}")
else:
    print("No content to analyze for offensiveness.")