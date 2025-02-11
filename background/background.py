from selenium import webdriver
from selenium.webdriver.common.by import By

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
    global browser
    browser = webdriver.Chrome(options=option, service=service)
    browser.get("https://www.artbreeder.com/tools/prompter")
    browser.implicitly_wait(30)


def get_images(prompts: list):
    init()

    generate_button = browser.find_element(By.XPATH, "//button[@class='generate-button']")

    text_area = browser.find_element(By.TAG_NAME ,"textarea")
    text_area.clear()

    for prompt in prompts:
        text_area.send_keys(prompt)
    
    while True:
        pass


if __name__ == "__main__":
    get_images(["a storyboard for a short film", "a storyboard for a short film"])