from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import requests
from time import sleep
import base64
import cloudscraper

driver_path = (
    r"e:\GitHub\Shorts-Reel-Generator\background\chromedriver-win64\chromedriver.exe"
)
brave_path = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"

option = webdriver.ChromeOptions()
option.binary_location = brave_path
option.add_argument("--no-sandbox")
option.add_argument("--disable-dev-shm-usage")
option.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
option.add_argument("user-data-dir='C:\\Users\\ryash\\AppData\\Local\\BraveSoftware\\Brave-Browser\\User Data\\Profile 2'")
# option.add_argument("--tor")
# option.add_argument("--headless")

service = webdriver.ChromeService(executable_path=driver_path)

browser: webdriver.Chrome = None


def init():
    global browser, option, service
    browser = webdriver.Chrome(options=option, service=service)
    # scraper = cloudscraper.create_scraper(browser)
    browser.implicitly_wait(30)
    # browser.get("https://www.artbreeder.com/tools/prompter")
    # browser.get("https://lorastudio.co/generate?model=artificialguybr/LineAniRedmond-LinearMangaSDXL-V2")
    # browser.get("https://lorastudio.co/generate?model=KappaNeuro/makoto-shinkai-style")
    browser.get("https://luvvoice.com/")


def quit_browser():
    if browser:
        browser.quit()


def get_images(prompts: list, prefix="", start=0):
    init()

    generate_button = browser.find_elements(By.TAG_NAME, "button")[-1]

    text_area = browser.find_element(By.TAG_NAME, "textarea")

    first_try = True

    for id, prompt in enumerate(prompts):
        while True:
            text_area.clear()
            text_area.send_keys(prompt)
            try:
                generate_button.click()
            except:
                continue

            if first_try:
                first_try = False
                sleep(7)
                break

            existing_images = browser.find_elements(By.TAG_NAME, "img")
            try:
                WebDriverWait(browser, 10).until(
                    lambda d: len(d.find_elements(By.TAG_NAME, "img"))
                    > len(existing_images)
                )
                sleep(1)
                break
            except:
                pass

        image = browser.find_element(By.TAG_NAME, "img")
        save_image(image.get_attribute("src"), prefix + str(id + start))

    browser.quit()


def get_bing_images(prompts: list, prefix="", start=0):
    init()

    generate_button = None
    for btn in browser.find_elements(By.TAG_NAME, "button"):
        if "Generate" in btn.text:
            generate_button = btn
            break

    text_area = browser.find_element(By.TAG_NAME, "textarea")

    for id, prompt in enumerate(prompts):
        print(f"[DEBUG] Generating image idx {id + start + 1} / {len(prompts)}")
        text_area.clear()
        text_area.send_keys(prompt)

        try:
            generate_button.click()
        except:
            print("Failed to click")
            browser.quit()
            exit(1)

        try:
            WebDriverWait(browser, 90).until(
                lambda d: d.find_element(By.XPATH, "//img[@alt='Generation']")
            )
        except:
            print("Failed to load image")
            print("Retrying!")
            browser.quit()

            get_bing_images(prompts[id:], prefix, start + id)

            return

        image = browser.find_element(By.XPATH, "//img[@alt='Generation']")
        image_data = image.get_attribute("src")
        image_data = image_data.split(",")[-1]
        image_data = base64.b64decode(image_data)
        with open(f"background/images/{prefix}{id + start}.png", "wb") as f:
            f.write(image_data)

    browser.quit()


initialised = False
generate_button = None
text_area = None

def set_intialised(val):
    global initialised
    initialised = val


def get_fast_images(prompts: list, prefix="", start=0):
    global initialised, generate_button, text_area

    if not initialised:
        init()
        initialised = True

        for btn in browser.find_elements(By.TAG_NAME, "button"):
            if "Generate" in btn.text:
                generate_button = btn
                break

        text_area = browser.find_element(By.TAG_NAME, "textarea")

    for id, prompt in enumerate(prompts):
        print(f"[DEBUG] Generating image idx {id + start + 1} / {len(prompts)}")
        text_area.clear()
        text_area.send_keys(prompt)

        try:
            generate_button.click()
        except:
            print("Failed to click")
            browser.quit()
            exit(1)

        try:
            WebDriverWait(browser, 90).until(
                lambda d: d.find_element(By.XPATH, "//img[@alt='Generation']")
            )
        except:
            print("Failed to load image")
            print("Retrying!")
            browser.quit()
            initialised = False

            get_fast_images(prompts[id:], prefix, start + id)

            return

        image = browser.find_element(By.XPATH, "//img[@alt='Generation']")
        image_data = image.get_attribute("src")
        image_data = image_data.split(",")[-1]
        image_data = base64.b64decode(image_data)
        with open(f"background/images/{prefix}{id + start}.png", "wb") as f:
            f.write(image_data)


def save_image(url: str, name: str):
    response = requests.get(url)
    response.raise_for_status()
    path = "background/images/" + name + ".jpeg"
    with open(path, "wb") as f:
        f.write(response.content)
    return path


if __name__ == "__main__":
    init()
    while True:
        pass
