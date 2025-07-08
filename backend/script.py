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

print("Extracted content:")
print(f"  User   : {text['user']}")
print(f"  Date   : {text['date']}")
print(f"  Content: {text['content']}")
print(f"  URL    : {text['url']}")

if text["content"] is not None:
    if ai_utils.is_offensive(text["content"]):
        print(ai_utils.classify_text(text["content"]))
    else:
        print("\nThe content is not offensive.")
else:
    print("\nNo content found in the provided URL.")