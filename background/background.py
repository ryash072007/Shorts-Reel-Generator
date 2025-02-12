from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import requests
from time import sleep

driver_path = (
    r"e:\GitHub\Shorts-Reel-Generator\background\chromedriver-win64\chromedriver.exe"
)
brave_path = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"

option = webdriver.ChromeOptions()
option.binary_location = brave_path
option.add_argument("--tor")
# option.add_argument("--headless")

service = webdriver.ChromeService(executable_path=driver_path)

browser: webdriver.Chrome = None


def init():
    global browser, option, service
    browser = webdriver.Chrome(options=option, service=service)
    browser.implicitly_wait(30)
    browser.get("https://www.artbreeder.com/tools/prompter")


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
                    lambda d: len(d.find_elements(By.TAG_NAME, "img")) > len(existing_images)
                )
                sleep(1)
                break
            except:
                pass

        image = browser.find_element(By.TAG_NAME, "img")
        save_image(image.get_attribute("src"), prefix+str(id + start))
    
    browser.quit()


def save_image(url: str, name: str):
    response = requests.get(url)
    response.raise_for_status()
    path = "background/images/" + name + ".jpeg"
    with open(path, "wb") as f:
        f.write(response.content)
    return path


if __name__ == "__main__":
    get_images(["a storyboard for a short film", "a storyboard for a dragon flying"])