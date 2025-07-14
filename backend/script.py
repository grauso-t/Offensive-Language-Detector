import tiktok
import threads
import ai_utils

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

# Get user input
url = input("Enter a TikTok or Threads URL: ").strip()
text = process_url(url)

if text is None or isinstance(text, str):
    print("No content found in the provided URL.")
    exit()

# Display extracted metadata
print("Extracted content:")
print(f"  User   : {text['user']}")
print(f"  Date   : {text['date']}")
print(f"  Content: {text['content']}")
print(f"  URL    : {text['url']}")

# Analyze content
if text["content"]:
    logistic_regression = ai_utils.is_offensive_logistic_regression(text["content"])
    svm = ai_utils.is_offensive_svm(text["content"])
    bert = ai_utils.is_offensive_bert(text["content"])
    gpt = ai_utils.is_offensive_gpt(text["content"])
    llm = ai_utils.is_offensive_llm(text["content"])

    # Print model results
    print("\n--- Offensiveness Analysis ---")
    print(f"Logistic Regression: {logistic_regression}")
    print(f"SVM                : {svm}")
    print(f"BERT               : {bert}")
    print(f"GPT-3.5            : {gpt}")
    print(f"Mistral:           : {llm}")

else:
    print("No content to analyze.")