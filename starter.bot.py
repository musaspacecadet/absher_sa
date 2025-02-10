from botasaurus.browser import browser, Driver
from model import create_inference_pipeline, model_path

# Create the inference pipeline
img_width = 200
img_height = 50

id_ = "1001527983"
phone = "543929796" 

fail_id = "1000389773"
fail_phone = "552050147"

predict_fn = create_inference_pipeline(model_path, img_width, img_height)
#<input type="submit" value="Cancel" id="viewns_Z7_GPHIG0G0LOORC0QS3PEBAI12R7_:validationForm:cancelBTN" name="viewns_Z7_GPHIG0G0LOORC0QS3PEBAI12R7_:validationForm:cancelBTN">
#<a href="#" onclick="window.location='https://absher.sa/wps/portal/individuals/Home/homepublic/!ut/p/z1/04_Sj9CPykssy0xPLMnMz0vMAfIjo8ziDTxNTDwMTYy83Q3MjAwcw4IsTFw9TQ3dzUz0wwkpiAJJ4wCOBkD9UViUOBo4BRk5GRsYuPsbYVWAYkZBboRBpqOiIgBIR9Vv/dz/d5/L2dBISEvZ0FBIS9nQSEh/'; return false;" style="padding:4px 25px !important" class="btnSubmit">Cancel</a>


def scrape_heading_task():
    driver = Driver(block_images_and_css=True, profile=False)
    page = driver.get("https://www.absher.sa/wps/vanityurl/en/resetpasswordindividuals")
    
    captcha_image = page.select("img.captchaImg")   
    captcha_input = page.select("input.captchaNum")
    userid = page.select("input.userNameId")
    mobile_number = page.select("input.mobileNumberText")
    submit_button = page.select("input.next")

    image_path = captcha_image.save_screenshot("captcha.png")

    predicted_text = predict_fn(image_path)
    print(f"Predicted CAPTCHA: {predicted_text}")
    captcha_input.send_keys(predicted_text)

    
    userid.send_keys(fail_id)
    mobile_number.send_keys(fail_phone)
    submit_button.click()

    header = page.wait_for('h1[tabindex="0"]', timeout=60)
    # Attempt to find the warning message element, if present
    warning_text = ('Sorry, the registered number is not activated please visit Absher self-service kiosk device or through '
                    'the General Directorate of Passport or Civil affairs offices .')
    fail = page.find_element_by_text(warning_text)

    # Extract header text (adjust method based on your API)
    header_text = header.get_html()
    
    print("Header:", header_text)
    if fail:
        print("Warning Message Detected:", warning_text)
    
    # Check authentication status based on header and warning message
    if "Security Validation" in header_text and not fail:
        print("Authentication Successful: 'Security Validation' found and no warning message detected.")
    elif "Reset Password" in header_text or fail:
        print("Authentication Failed: Either 'Reset Password' found in header or warning message detected.")
    else:
        print(f"Authentication status unclear. Header text: {header_text}")

    driver.delete_cookies_and_local_storage()
    driver.get("https://www.absher.sa/wps/vanityurl/en/resetpasswordindividuals")
    # you can start again here 
    driver.close()
    

# Initiate the web scraping task
scrape_heading_task()