from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup

def extract_html_div(url):
    if not url:
        print("URL not provided.")
        return
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(url)
        time.sleep(3)
        
        div_elements = driver.find_elements(By.XPATH,
            "//div[contains(concat(' ', normalize-space(@class), ' '), ' xrvj5dj ') and "
            "contains(concat(' ', normalize-space(@class), ' '), ' xd0jker ') and "
            "contains(concat(' ', normalize-space(@class), ' '), ' x11ql9d ') and "
            "contains(concat(' ', normalize-space(@class), ' '), ' x1e9u5ye ')]"
        )
        
        if div_elements:
            html = div_elements[0].get_attribute("outerHTML")
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract user with error handling
            user_element = soup.find("span", class_="x1lliihq x193iq5w x6ikm8r x10wlt62 xlyipyv xuxw1ft")
            user = user_element.text if user_element else None
            
            # Extract date with error handling
            date_element = soup.find("time", class_="x1rg5ohu xnei2rj x2b8uid xuxw1ft")
            date = date_element.text if date_element else None
            
            # Extract content with error handling
            content_span = soup.find("span", class_="x1lliihq x1plvlek xryxfnj x1n2onr6 x1ji0vk5 x18bv5gf xi7mnp6 x193iq5w xeuugli x1fj9vlw x13faqbe x1vvkbs x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x1i0vuye xjohtrz xo1l8bm xp07o12 x1yc453h xat24cr xdj266r")
            content = None
            if content_span:
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
        driver.quit()

# Execution
if __name__ == "__main__":
    url = input("Enter the URL of the page to extract: ")
    result = extract_html_div(url)
    
    if result:
        print(f"Extraction result: {result}")