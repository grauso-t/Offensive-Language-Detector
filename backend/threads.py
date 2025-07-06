from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup

def is_threads_url(url):
    """Check if the URL belongs to Threads"""
    try:
        parsed = urlparse(url)
        threads_domains = [
            'threads.com',
            'www.threads.com'
        ]
        return parsed.netloc.lower() in threads_domains
    except Exception:
        return False

def parse_threads(url):
    """
    Parse a Threads post URL and extract user, date, and content information.
    
    Args:
        url (str): The Threads post URL to parse
        
    Returns:
        dict: Dictionary containing user, date, and content information
    """
    if not url:
        print("URL not provided.")
        return {
            'user': None,
            'date': None,
            'content': None
        }
    
    # Configure Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run browser in background
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    
    # Initialize Chrome service
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Navigate to the Threads URL
        driver.get(url)
        time.sleep(3)  # Wait for page to load
        
        # Find the main post container using XPath with specific CSS classes
        div_elements = driver.find_elements(By.XPATH,
            "//div[contains(concat(' ', normalize-space(@class), ' '), ' xrvj5dj ') and "
            "contains(concat(' ', normalize-space(@class), ' '), ' xd0jker ') and "
            "contains(concat(' ', normalize-space(@class), ' '), ' x11ql9d ') and "
            "contains(concat(' ', normalize-space(@class), ' '), ' x1e9u5ye ')]"
        )
        
        if div_elements:
            # Get the HTML content of the first matching element
            html = div_elements[0].get_attribute("outerHTML")
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract username with error handling
            user_element = soup.find("span", class_="x1lliihq x193iq5w x6ikm8r x10wlt62 xlyipyv xuxw1ft")
            user = user_element.text if user_element else None
            
            # Extract post date with error handling
            date_element = soup.find("time", class_="x1rg5ohu xnei2rj x2b8uid xuxw1ft")
            date = date_element.text if date_element else None
            
            # Extract post content with error handling
            content_span = soup.find("span", class_="x1lliihq x1plvlek xryxfnj x1n2onr6 x1ji0vk5 x18bv5gf xi7mnp6 x193iq5w xeuugli x1fj9vlw x13faqbe x1vvkbs x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x1i0vuye xjohtrz xo1l8bm xp07o12 x1yc453h xat24cr xdj266r")
            content = None
            if content_span:
                # Look for inner span containing the actual text content
                inner_span = content_span.find('span')
                content = inner_span.get_text(strip=True) if inner_span else None
            
            return {
                'user': user,
                'date': date,
                'content': content
            }
        
        else:
            print("No element found with the specified selector.")
            return {
                'user': None,
                'date': None,
                'content': None
            }
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return {
            'user': None,
            'date': None,
            'content': None
        }
    
    finally:
        # Always close the browser driver to free up resources
        driver.quit()

def main():
    """Main function to test the Threads parser."""
    
    # Example usage
    print("🧵 Threads Post Parser")
    print("=" * 25)
    
    # Test URL (replace with actual Threads URL)
    test_url = "https://www.threads.net/@username/post/example"
    
    if is_threads_url(test_url):
        print(f"Valid Threads URL: {test_url}")
        result = parse_threads(test_url)
        
        print("\nParsed Results:")
        print(f"User: {result['user']}")
        print(f"Date: {result['date']}")
        print(f"Content: {result['content']}")
    else:
        print(f"Invalid Threads URL: {test_url}")

if __name__ == "__main__":
    main()