# import json
# import time
# import re
# import pandas as pd
# import random
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By

# # --- CONFIGURATION ---
# TARGET_CAR_COUNT = 20  # How many cars do you want to scrape?
# CSV_FILENAME = "cars_and_bids_final.csv"

# def setup_driver():
#     chrome_options = Options()
#     # Looks like a real user
#     chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
#     chrome_options.add_argument("--disable-blink-features=AutomationControlled")
#     # chrome_options.add_argument("--headless") # Uncomment if you don't want to see the browser
#     return webdriver.Chrome(options=chrome_options)

# def get_auction_urls(driver, limit=20):
#     """Scrolls the 'Past Auctions' page and collects valid auction URLs."""
#     print(f"--- Phase 1: Collecting {limit} URLs ---")
#     driver.get("https://carsandbids.com/past-auctions/")
#     time.sleep(3)

#     collected_urls = set()
    
#     # Regex to ensure we only get valid auction links (avoids 'create listing' or broken links)
#     # Valid format: /auctions/{id}/{slug}
#     url_pattern = re.compile(r'/auctions/[A-Za-z0-9]{8,}/.+')

#     while len(collected_urls) < limit:
#         # Scroll down
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(random.uniform(1.5, 3)) # Random wait to look human
        
#         # Find all links
#         links = driver.find_elements(By.TAG_NAME, "a")
        
#         for link in links:
#             href = link.get_attribute('href')
#             if href and url_pattern.search(href):
#                 collected_urls.add(href)
        
#         print(f"Found {len(collected_urls)} unique cars so far...")
        
#         # If we have enough, stop scrolling
#         if len(collected_urls) >= limit:
#             break
            
#     return list(collected_urls)[:limit]

# def extract_car_data(driver, url):
#     """
#     Extracts data using the hidden JSON object (Primary) 
#     and falls back to XPaths (Secondary).
#     """
#     driver.get(url)
#     time.sleep(random.uniform(2, 4))
    
#     data = {"URL": url}
#     try:
#         data['Method'] = 'XPath'
        
#         # 1. Price
#         try:
#             # Your specific XPath for price
#             #price_xpath = "/html/body/div[4]/div[2]/div[6]/div/div[8]/div/div/span/span"
#             price_xpath = "/html/body/div[4]/div[2]/div[6]/div/div[8]/div/div/span"
#             data['Sold_Price'] = driver.find_element(By.XPATH, price_xpath).text
#         except:
#             pass

#         # 2. Quick Facts Table
#         try:
#             # Helper to find value by label text
#             def get_fact(label):
#                 # XPath: Find 'dt' with text, go to following 'dd'
#                 return driver.find_element(By.XPATH, f"//dt[contains(text(), '{label}')]/following-sibling::dd").text

#             data['Make'] = get_fact("Make")
#             data['Model'] = get_fact("Model")
#             data['Mileage'] = get_fact("Mileage")
#             # data['VIN'] = get_fact("VIN")
#             data['Title Status'] = get_fact("Title Status")
#             # data['Location'] = get_fact("Location")
#             data['Seller Type'] = get_fact("Seller Type")
#             data['Engine'] = get_fact("Engine")
#             data['Drivetrain'] = get_fact("Drivetrain")
#             data['Transmission'] = get_fact("Transmission")
#             data['Body Style'] = get_fact("Body Style")
#             data['Exterior Color'] = get_fact("Exterior Color")
#             data['Interior Color'] = get_fact("Interior Color")
            
#         except:
#             pass
            
#     except Exception as e:
#         print(f"Visual scrape failed: {e}")

#     return data

# def main():
#     driver = setup_driver()
#     all_cars = []
    
#     try:
#         # 1. Collect URLs
#         urls = get_auction_urls(driver, limit=TARGET_CAR_COUNT)
        
#         print(f"\n--- Phase 2: Extracting Data from {len(urls)} Cars ---")
        
#         # 2. Extract Data
#         for i, url in enumerate(urls):
#             print(f"[{i+1}/{len(urls)}] Processing: {url}")
#             try:
#                 car_data = extract_car_data(driver, url)
#                 if car_data:
#                     all_cars.append(car_data)
#                     # Simple progress print
#                     price = car_data.get('Sold_Price') or car_data.get('High_Bid')
#                     print(f"   -> Success: {car_data.get('Title')} | ${price}")
#             except Exception as e:
#                 print(f"   -> Error: {e}")

#     finally:
#         driver.quit()

#     # 3. Save
#     if all_cars:
#         df = pd.DataFrame(all_cars)
#         df.to_csv(CSV_FILENAME, index=False)
#         print(f"\nSUCCESS: Saved {len(df)} cars to '{CSV_FILENAME}'")
#     else:
#         print("\nNo data collected.")

# if __name__ == "__main__":
#     main()

# import time
# import re
# import pandas as pd
# import random
# import os
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # --- CONFIGURATION ---
# CSV_FILENAME = "cars_and_bids_full_history.csv"

# def setup_driver():
#     chrome_options = Options()
#     # Looks like a real user to avoid being blocked
#     chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
#     chrome_options.add_argument("--disable-blink-features=AutomationControlled")
#     # chrome_options.add_argument("--headless") # Uncomment to run invisibly
#     return webdriver.Chrome(options=chrome_options)

# def get_all_auction_urls(driver):
#     """
#     Loops through all pages (1, 2, 3...) clicking 'Next' until the end.
#     """
#     print(f"--- Phase 1: Collecting ALL URLs ---")
#     driver.get("https://carsandbids.com/past-auctions/")
#     time.sleep(3)

#     ordered_urls = [] # List keeps order (Newest -> Oldest)
#     seen_urls = set() # Set prevents duplicates
    
#     url_pattern = re.compile(r'/auctions/[A-Za-z0-9]{8,}/.+')
#     page_num = 1
    
#     while True:
#         print(f"Scanning Page {page_num}...")
        
#         # 1. Collect links on the CURRENT page
#         links = driver.find_elements(By.TAG_NAME, "a")
#         new_on_page = 0
        
#         for link in links:
#             href = link.get_attribute('href')
#             if href and url_pattern.search(href):
#                 if href not in seen_urls:
#                     seen_urls.add(href)
#                     ordered_urls.append(href)
#                     new_on_page += 1
        
#         print(f"   -> Found {new_on_page} cars on this page. Total unique: {len(ordered_urls)}")

#         # 2. Try to find and click "Next"
#         try:
#             # Look for the Next button. 
#             # Usually it's an 'li' with class 'next' or an arrow icon.
#             # We try a few common XPaths for robustness.
#             next_button_xpath = "//li[contains(@class, 'next') and not(contains(@class, 'disabled'))]/a"
            
#             # Wait briefly to ensure button is present
#             next_btn = driver.find_element(By.XPATH, next_button_xpath)
            
#             # Scroll it into view just in case
#             driver.execute_script("arguments[0].scrollIntoView();", next_btn)
#             time.sleep(1) 
            
#             next_btn.click()
            
#             # Wait for the next page to load
#             time.sleep(random.uniform(2, 4))
#             page_num += 1
            
#         except Exception:
#             # If we can't find the Next button (or it's disabled), we are done.
#             print("\nReached the last page (no 'Next' button found).")
#             break
            
#     return ordered_urls

# def extract_car_data(driver, url):
#     """
#     Extracts data using the hidden JSON object (Primary) 
#     and falls back to XPaths (Secondary).
#     """
#     driver.get(url)
#     time.sleep(random.uniform(1.5, 3))
    
#     data = {"URL": url}
#     try:
#         # 1. Price
#         # try:
#         #     # Generic approach to find price on the page
#         #     price_xpaths = [
#         #         "/html/body/div[4]/div[2]/div[6]/div/div[8]/div/div/span",
#         #         "//span[@class='bid-value']",
#         #         "//div[@class='auction-heading']//span[@class='value']"
#         #     ]
#         #     for xpath in price_xpaths:
#         #         try:
#         #             price_element = driver.find_element(By.XPATH, xpath)
#         #             data['Sold_Price'] = price_element.text
#         #             break
#         #         except:
#         #             continue
#         # except:
#         #     pass
#         try:
#             # Your specific XPath for price
#             #price_xpath = "/html/body/div[4]/div[2]/div[6]/div/div[8]/div/div/span/span"
#             price_xpath = "/html/body/div[4]/div[2]/div[6]/div/div[8]/div/div/span"
#             data['Sold_Price'] = driver.find_element(By.XPATH, price_xpath).text
#         except:
#             pass

#         # 2. Quick Facts Table
#         try:
#             def get_fact(label):
#                 return driver.find_element(By.XPATH, f"//dt[contains(text(), '{label}')]/following-sibling::dd").text

#             # Standard fields
#             fields = ["Make", "Model", "Mileage", "Title Status", "Seller Type", 
#                       "Engine", "Drivetrain", "Transmission", "Body Style", 
#                       "Exterior Color", "Interior Color"]
            
#             for field in fields:
#                 try:
#                     data[field] = get_fact(field)
#                 except:
#                     data[field] = None
#         except:
#             pass
            
#     except Exception as e:
#         print(f"Visual scrape failed: {e}")

#     return data

# def save_single_row(data, filename):
#     """Appends a single row of data to the CSV immediately."""
#     df = pd.DataFrame([data])
#     if not os.path.isfile(filename):
#         df.to_csv(filename, index=False)
#     else:
#         df.to_csv(filename, mode='a', header=False, index=False)

# def main():
#     driver = setup_driver()
    
#     try:
#         # 1. Collect ALL URLs (Page by Page)
#         urls = get_all_auction_urls(driver)
        
#         print(f"\n--- Phase 2: Extracting Data from {len(urls)} Cars ---")
        
#         # 2. Extract Data
#         for i, url in enumerate(urls):
#             print(f"[{i+1}/{len(urls)}] Processing: {url}")
#             try:
#                 car_data = extract_car_data(driver, url)
#                 if car_data:
#                     # SAVE IMMEDIATELY
#                     save_single_row(car_data, CSV_FILENAME)
                    
#                     # Progress Print
#                     price = car_data.get('Sold_Price') or "N/A"
#                     title = f"{car_data.get('Make', '')} {car_data.get('Model', '')}"
#                     print(f"   -> Saved: {title} | {price}")
#             except Exception as e:
#                 print(f"   -> Error: {e}")

#     finally:
#         driver.quit()
#         print(f"\nFinished. Data saved to {CSV_FILENAME}")

# if __name__ == "__main__":
#     main()

import time
import re
import pandas as pd
import random
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIGURATION ---
CSV_FILENAME = "cars_and_bids_full_history.csv"

def setup_driver():
    chrome_options = Options()
    # Looks like a real user to avoid being blocked
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    # chrome_options.add_argument("--headless") # Uncomment to run invisibly
    return webdriver.Chrome(options=chrome_options)

def get_all_auction_urls(driver):
    """
    Loops through all pages (1, 2, 3...) clicking 'Next' until the end.
    """
    print(f"--- Phase 1: Collecting ALL URLs ---")
    driver.get("https://carsandbids.com/past-auctions/")
    time.sleep(3)

    ordered_urls = [] # List keeps order (Newest -> Oldest)
    seen_urls = set() # Set prevents duplicates
    
    url_pattern = re.compile(r'/auctions/[A-Za-z0-9]{8,}/.+')
    page_num = 1
    
    while True:
        print(f"Scanning Page {page_num}...")
        
        # 1. Collect links on the CURRENT page
        # Wait for at least one auction link to be present to ensure page load
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/auctions/')]"))
            )
        except:
            print("   -> No auction links found on this page.")
        
        links = driver.find_elements(By.TAG_NAME, "a")
        new_on_page = 0
        
        for link in links:
            href = link.get_attribute('href')
            if href and url_pattern.search(href):
                if href not in seen_urls:
                    seen_urls.add(href)
                    ordered_urls.append(href)
                    new_on_page += 1
        
        print(f"   -> Found {new_on_page} cars on this page. Total unique: {len(ordered_urls)}")

        # 2. Try to find and click "Next"
        try:
            # ROBUST NEXT BUTTON SELECTOR:
            # Tries to find the button by Text ("Next"), Aria Label, or Class.
            next_btn = None
            
            potential_xpaths = [
                "//a[contains(text(), 'Next')]",             # Standard text link
                "//button[contains(text(), 'Next')]",        # Standard button
                "//li[contains(@class, 'next')]/a",          # Old style li > a
                "//a[contains(@class, 'next')]",             # Class on a tag
                "//a[@aria-label='Next page']",              # Accessibility label
                "//ul[contains(@class, 'pagination')]//li[last()]/a" # Last item in pagination list
            ]
            
            for xpath in potential_xpaths:
                try:
                    element = driver.find_element(By.XPATH, xpath)
                    if element.is_displayed() and element.is_enabled():
                        next_btn = element
                        break
                except:
                    continue
            
            if next_btn:
                # Scroll into view to avoid "element click intercepted"
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
                time.sleep(1) # Small pause after scroll
                
                # JavaScript click is often more reliable for pagination buttons overlayed by footers
                driver.execute_script("arguments[0].click();", next_btn)
                
                # Wait for the page number to change or a brief pause
                time.sleep(random.uniform(2, 4))
                page_num += 1
            else:
                print("\nReached the last page (no 'Next' button found).")
                break
                
        except Exception as e:
            print(f"\nError trying to click next: {e}")
            break
            
    return ordered_urls

def extract_car_data(driver, url):
    """
    Extracts data using the hidden JSON object (Primary) 
    and falls back to XPaths (Secondary).
    """
    driver.get(url)
    time.sleep(random.uniform(1.5, 3))
    
    data = {"URL": url}
    try:
        try:
            # Your specific XPath for price
            price_xpath = "/html/body/div[4]/div[2]/div[6]/div/div[8]/div/div/span"
            data['Sold_Price'] = driver.find_element(By.XPATH, price_xpath).text
        except:
            pass

        # 2. Quick Facts Table
        try:
            def get_fact(label):
                return driver.find_element(By.XPATH, f"//dt[contains(text(), '{label}')]/following-sibling::dd").text

            # Standard fields
            fields = ["Make", "Model", "Mileage", "Title Status", "Seller Type", 
                      "Engine", "Drivetrain", "Transmission", "Body Style", 
                      "Exterior Color", "Interior Color"]
            
            for field in fields:
                try:
                    data[field] = get_fact(field)
                except:
                    data[field] = None
        except:
            pass
            
    except Exception as e:
        print(f"Visual scrape failed: {e}")

    return data

def save_single_row(data, filename):
    """Appends a single row of data to the CSV immediately."""
    df = pd.DataFrame([data])
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

def main():
    driver = setup_driver()
    
    try:
        # 1. Collect ALL URLs (Page by Page)
        urls = get_all_auction_urls(driver)
        
        print(f"\n--- Phase 2: Extracting Data from {len(urls)} Cars ---")
        
        # 2. Extract Data
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Processing: {url}")
            try:
                car_data = extract_car_data(driver, url)
                if car_data:
                    # SAVE IMMEDIATELY
                    save_single_row(car_data, CSV_FILENAME)
                    
                    # Progress Print
                    price = car_data.get('Sold_Price') or "N/A"
                    title = f"{car_data.get('Make', '')} {car_data.get('Model', '')}"
                    print(f"   -> Saved: {title} | {price}")
            except Exception as e:
                print(f"   -> Error: {e}")

    finally:
        driver.quit()
        print(f"\nFinished. Data saved to {CSV_FILENAME}")

if __name__ == "__main__":
    main()