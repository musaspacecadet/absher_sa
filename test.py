from botasaurus.browser import browser, Driver
from chrome_extension_python import Extension

Driver.block_images_and_css
@browser(
    block_images_and_css=True
)
def scrape_while_blocking_ads(driver: Driver, data):
    driver.prompt()

scrape_while_blocking_ads()