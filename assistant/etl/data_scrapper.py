import os
import csv
import time
import re
from turtle import title
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from webdriver_manager.core.os_manager import OperationSystemManager, ChromeType
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import ast

class FlipkartScrapper:
    """
    Class scrapes products and reviews from Flipkart based on a search query and stores the results in a CSV file.
    """
    # Product Title: <a> with data-tracking attribute
    TITLE_SELECTORS = [
        "[data-tracking-id]",          # data attribute on title links
        "a[title]",                    # <a title="product name">
        "a[aria-label]",               # accessible label carries the name
        "h1", "h2", "h3",              # semantic headings inside the card
    ]
    # Price: look for elements that carry currency symbols in their text
    PRICE_SELECTOR = [
        "[data-testid='price']",
        "[itemprop='price']",
        "[aria-label*='price' i]",
        "[aria-label*='₹' i]",
    ]
    # Rating: star / numeric rating elements typically carry aria or data hints.
    RATING_SELECTORS = [
        "[aria-label*='rating' i]",
        "[data-testid='rating']",
        "[itemprop='ratingValue']",
        "span[role='img'][aria-label]",
    ]
    # Review count: elements that mention "Ratings" or "Reviews" in their text
    REVIEW_COUNT_SELECTORS = [
        "[itemprop='reviewCount']",
        "[itemprop='ratingCount']",
        "[data-testid*='review' i]",
        "[aria-label*='review' i]",
        "[aria-label*='rating' i]",
    ]

    # Product link page: any <a> whose href contains canonical product path
    PRODUCT_LINK_SELECTOR = "a[href*='/p/']"

    # Review blocks on a product detail page - semantic selectors.
    REVIEW_BLOCK_SELECTOR = [
        "[itemprop='review']",
        "[itemprop='reviewBody']",
        "[data-testid='review-block']",
        "[data-testid='review']",
        "[data-review-id]",
        "[data-review]",
        "[aria-label*='review' i]",
        "[role='article'][aria-label*='review' i]",
        "article",
        "article > p",
        "[id*='review' i]",
        "[class*='review' i]",
    ]

    def __init__(self, outputdir: str="data"):
        self.outputdir = outputdir
        os.makedirs(self.outputdir, exist_ok=True)

    # Internal helpers

    def _make_driver(self):
        """ Create and return a configured undetected chromedriver instance. """
        br_ver = OperationSystemManager().get_browser_version_from_os(ChromeType.GOOGLE)
        version_main = int(br_ver.split('.')[0])
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver=uc.Chrome(options=options, use_subprocess=True, version_main=version_main)
        return driver
    
    #staticmethod
    def _dismiss_popup(self,driver) -> None:
        """ Try to close flipkart login/signup pop-up if it appears. """
        try:
            close_btn = None
            for xpath in [
                "//button[@aria-label='close' or @aria-label='Close']",
                "//button[normalize-space(text())='x' or normalize-space(text())='X']",
                "//button[contains(@class, 'close')]",
            ]:
                results = driver.find_elements(By.XPATH, xpath)
                if results:
                    close_btn = results[0]
                    break
            if close_btn:
                close_btn.click()
                time.sleep(1)
        except Exception as e:
            print(f"[dismiss popup] Could not close pop-up: {e}")
    
    
    @staticmethod
    def _first_text(element, selectors: list[str]) -> str:
        """
        Try each CSS selector in order against *element* and return the stripped
        text of the first matching child. Falls back to '' if nothing matches.
        """
        print(f"Trying to extract text using selectors: {selectors}")
        for sel in selectors:
            try:
                found = element.find_element(By.CSS_SELECTOR, sel)
                print(f"Found element for selector '{sel}': {found.tag_name}")
                text = found.text.strip()
                print(f"Trying selector '{sel}' found text: '{text}'")
                if text:
                    return text
                for attr in ("aria-label", "data-value", "content", "title"):
                    val = found.get_attribute(attr)
                    print(f"Checking attribute '{attr}' for text: {val}")
                    if val and val.strip():
                        return val.strip()
            except Exception:
                continue
        return ""
    
    @staticmethod
    def _extract_title_from_url(url: str) -> str:
        """
        Flipkart URLs follow: /product-name-with-hyphens/p/pid
        Extract and clean the product name segment.
        """
        try:
            path = url.split("flipkart.com/")[-1]   # e.g. apple-iphone-16-black-128-gb/p/itm...
            slug = path.split("/")[0]                 # apple-iphone-16-black-128-gb
            return slug.replace("-", " ").title()     # Apple Iphone 16 Black 128 Gb
        except Exception:
            return ""

    @staticmethod
    def _extract_price(element) -> str:
        """ 
        Look for a price in *element* by:
        1. Trying semantic selectors.
        2. Falling back to regex scan of the element's full text.
        """
        for sel in FlipkartScrapper.PRICE_SELECTOR:
            try:
                found = element.find_element(By.CSS_SELECTOR, sel)
                text = found.text.strip()
                if "₹" in text:
                    return text
            except Exception:
                continue
        # Fallback: regex scan for price patterns in the element's full text
        match = re.search(r"₹\s?[\d,]+", element.text)
        return match.group(0) if match else "N/A"
    
    @staticmethod
    def _extract_rating(item) -> str:
        # Try direct text of any span/div that looks like a rating (single decimal)
        candidates = item.find_elements(By.CSS_SELECTOR, "span, div")
        for el in candidates:
            try:
                text = el.text.strip()
                if re.fullmatch(r"[1-5]\.[0-9]", text):   # e.g. "4.3"
                    return text
            except Exception:
                continue
        return "N/A"

    @staticmethod
    def _extract_review_count(element) -> str:
        for sel in FlipkartScrapper.REVIEW_COUNT_SELECTORS:
            try:
                found = element.find_element(By.CSS_SELECTOR, sel)
                for source in (found.text, found.get_attribute("aria-label"),
                               found.get_attribute("content")):
                    if source:
                        match = re.search(r"[\d,]+", source)
                        if match:
                            return match.group(0).replace(",", "")
            except Exception:
                continue
        match = re.search(r"([\d,]+)\s+(?:Ratings?|Reviews?)", element.text, re.I)
        return match.group(1).replace(",", "") if match else "N/A"
    
    @staticmethod
    def _extract_product_link(element) -> str:
        try:
            link_elem = element.find_element(By.CSS_SELECTOR, FlipkartScrapper.PRODUCT_LINK_SELECTOR)
            href = link_elem.get_attribute("href")
            if href.startswith("http"):
                return href.split("?")[0]  # Remove tracking params
        except Exception:
            pass
        return ""
    
    @staticmethod
    def _extract_product_id(url: str) -> str:
        match = re.findall(r"/p/(itm[0-9a-zA-Z]+)", url)
        return match[0] if match else "N/A"
    
    @staticmethod
    def _construct_review_link(url: str) -> str:
        match = re.findall(r"/p/(itm[0-9a-zA-Z]+)", url)
        prod_id = match[0] if match else None
        return url.replace(f"/p/{prod_id}", f"/product-reviews/{prod_id}") if prod_id else "N/A"

    # Review scrapping

    def _is_noise_block(self, text: str) -> bool:
        """ Filter out non-review blocks checks for noise signals. """
        #text = tag.get_text(strip=True).lower()
        noise_keywords = [
            "add to cart", "buy now", "offers", "specifications",
            "highlights", "seller", "exchange", "emi", "delivery",
            "contact us", "about us", "careers", "privacy", "sitemap", 
            "registered office", "cin :", "telephone:", "grievance",
            "terms of use", "cancellation & returns",
        ]
        return any(kw in text.lower() for kw in noise_keywords)
    
    def _is_valid_review_text(self, text: str) -> bool:
        """
        A valid review should be :
         - Be within reasonable length limits
         - Have enough words to form a sentence/opinion
         - Not be a single long run-on (footer/address form)
        """ 
        words = text.split()
        word_count = len(words)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)

        return (
            40 < len(text) < 2000 and
            8 <= word_count <= 300 and
            avg_word_length < 12
        )
    
    def _has_noise_ancestor(self, tag) -> bool:
        """
        Walk up the DOM tree - if any ancestor is a footer, nav, header, or sidebar, 
        this block is structural content and not a review.
        """
        NOISE_ANCESTORS = {"footer", "nav", "header", "aside", "form", "table"}
        for parent in tag.parents:
            if parent.name in NOISE_ANCESTORS:
                return True
            role = parent.get("role", "")
            el_id = parent.get("id", "")
            el_class = " ".join(parent.get("class", [])).lower()
            if role in ("navigation", "banner", "contentinfo"):
                return True
            if any(kw in el_id or kw in el_class for kw in ("footer", "navbar", "sidebar", "breadcrumb", "header")):
                return True
        return False


#     def get_top_reviews(self, product_url: str, count: int=2) -> str:
#         """
#         Navigate to the product page and get upto *count* top review texts.
#         Uses structural/semantic selectors rather than hard-coded class names.
#         """
#         if not product_url.startswith("http"):
#             return "No reviews found"
        
#         driver = self._make_driver()
#         reviews: list[str] = []

#         try:
#             driver.get(product_url)
#             time.sleep(4)
#             self._dismiss_popup(driver)

#             for _ in range(6):
#                 ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform()
#                 time.sleep(2)
            
#             soup = BeautifulSoup(driver.page_source, "html.parser")
#             seen: set[str] = set()

#             # --- Strategy 1: semantic / data-* selectors ---
#             for sel in self.REVIEW_BLOCK_SELECTOR:
#                 blocks = soup.select(sel)
#                 for block in blocks:
#                     if self._is_noise_block(block) or self._has_noise_ancestor(block):
#                         continue
                    
#                     text = block.get_text(separator=" ", strip=True)
#                     if (
#                         self._is_valid_review_text(text)
#                         and text not in seen
#                     ):
# #                    len(text) > 40 and text not in seen and len(text) < 2000 and not self._is_noise_block(block):
#                         reviews.append(text)
#                         seen.add(text)
#                     if len(reviews) >= count:
#                         break
#                 if len(reviews) >= count:
#                     break

#                 # -- Strategy 2: structural heuristics --
#                 if not reviews:
#                     candidates = soup.find_all(
#                         lambda tag: tag.name in ("div", "section")
#                         and tag.find("p")
#                         # and 80 < len(tag.get_text(strip=True)) < 1500
#                         and not tag.find(["nav", "header", "footer", "table"])
#                     )
#                     for cand in candidates:
#                         if self._is_noise_block(cand) or self._has_noise_ancestor(cand):
#                             continue

#                         text = cand.get_text(separator=" ", strip=True)
#                         if self._is_valid_review_text(text) and text not in seen:
# #                        if text not in seen and not self._is_noise_block(cand):
#                             reviews.append(text)
#                             seen.add(text)
                        
#                         if len(reviews) >= count:
#                             break
        
#         except Exception as e:
#             print(f"[get_top_reviews] Error: {e}")
#         finally:
#             driver.quit()
        
#         return "||".join(reviews) if reviews else "No reviews found"

    def get_top_reviews(self, product_url: str, count: int = 2) -> str:
        """
        Navigate to a product page and extract up to *count* review texts.
        Uses structural/semantic selectors rather than hard-coded class names.
        """
        if not product_url.startswith("http"):
            return "No reviews found"
        
        print(f"[get_top_reviews] Fetching reviews from: {product_url}")

        driver = self._make_driver()
        reviews: list[str] = []

        try:
            driver.get(product_url)
            time.sleep(4)
            self._dismiss_popup(driver)

            for _ in range(4):
                ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform()
                time.sleep(1.5)

            soup = BeautifulSoup(driver.page_source, "html.parser").get_text(separator="\n", strip=True)
            review_pattern = re.compile(
                r"\d+\.\d+\s*[•·]\s*.+?\n"
                r"Review for:.+?\n"
                r"(.+?)\n"
                r".+?(?:Verified Purchase|Gold Reviewer)",
                re.DOTALL
            )

            for match in review_pattern.finditer(soup):
                review_body = match.group(1).strip()
                print(f"[get_top_reviews] Matched review text: {match.group(0)[:100]}... {match.group(1)[:100]}...")
                if (
                    len(review_body) > 5 and not self._is_noise_block(review_body)
                ):
                    reviews.append(review_body)
                if len(reviews) >= count:
                    break

        except Exception as exc:
            print(f"[get_top_reviews] Error: {exc}")
        finally:
            driver.quit()

        print(f"[get_top_reviews] Extracted {reviews} reviews.")

        return "||".join(reviews) if reviews else "No reviews found"
    
    # Product listing scraping

    def scrape_flipkart_products(self, query: str, max_products: int =1, review_count: int =2) -> list[list]:
        """
        Search Flipkart for *query* and collect up to *max_products* results,
        each with up to *review_count* reviews.
        """
        driver = self._make_driver()
        search_url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
        driver.get(search_url)
        time.sleep(4)
        self._dismiss_popup(driver)
        time.sleep(2)

        products: list[list] = []

        items = driver.find_elements(By.CSS_SELECTOR, "[data-id]")
        if not items:
            items = driver.find_elements(
                By.CSS_SELECTOR, "li[class], article[class], div[class][data-tkid]"
            )
        items = items[:max_products]
        print(f"[scrape_flipkart_products] Found {len(items)} product cards for query '{query}'.")

        for item in items:
            try:
                price = self._extract_price(item)
                rating = self._extract_rating(item)
                print(f"[scrape_flipkart_products] Extracted - Rating: {rating}")
                total_reviews = self._extract_review_count(item)
                print(f"[scrape_flipkart_products] Extracted - Reviews: {total_reviews}")
                product_link = self._extract_product_link(item)
                print(f"[scrape_flipkart_products] Extracted - Link: {product_link}")
                product_id = self._extract_product_id(product_link)
                print(f"[scrape_flipkart_products] Extracted - {product_id}")
                title = self._first_text(item, self.TITLE_SELECTORS)
                print(f"[scrape_flipkart_products] Extracted - Title: {title}")
                review_link = self._construct_review_link(product_link)
                print(f"[scrape_flipkart_products] Constructed review link: {review_link}")

                if not title and "flipkart.com" in product_link:
                    title = self._extract_title_from_url(product_link)
                    print(f"[scrape_flipkart_products] Title from URL fallback: {title}")

                if not title:
                    print("[scrape_flipkart_products] Skipping card - could not extract title.")
                    continue

            except Exception as e:
                print(f"[scrape_flipkart_products] Error extracting product info: {e}")
                continue

            top_reviews = (
                self.get_top_reviews(review_link, count=review_count) 
                if "flipkart.com" in review_link
                else "Invalid product URL"
            )
            print(f"[scrape_flipkart_products] Extracted top reviews: {top_reviews}")
            products.append([product_id, title, rating, total_reviews, price, top_reviews])
        
        driver.quit()
        return products
    
    # CSV Export
    def save_to_csv(self, data, filename: str = "product_reviews.csv") -> str:
        """ Save scraped data to a CSV file. Returns the resolved file path. """
        print(f"[save_to_csv] data type: {type(data)}")
        # if isinstance(data[[0]], list):
        #     data[[0]] = ast.literal_eval(data[[0]])
        #     print(f"[save_to_csv] literal eval: {data[[0]]}")

        if os.path.isabs(filename) or os.path.dirname(filename):
            path = filename
        else: path = os.path.join(self.outputdir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"]) 
            for row in data:
                writer.writerow(str(field) for field in row)
                print(f"[save_to_csv] Saved {(str(field) for field in row)} rows")

        return path
