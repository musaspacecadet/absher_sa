import asyncio
import nodriver as uc
from model import create_inference_pipeline, model_path
import base64
import uuid
import os

def save_image_from_base64(base64_string, output_dir="images"):
    """
    Decodes a base64 string and saves it as an image file with a unique name.
    
    Args:
        base64_string (str): The base64-encoded string representing the image.
        output_dir (str, optional): Directory to save the image. Defaults to "images".
    
    Returns:
        str: The file path to the saved image.
    """
    # Ensure the output directory exists; create it if it doesn't.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate a unique filename with a .png extension
    unique_filename = f"{uuid.uuid4()}.png"
    file_path = os.path.join(output_dir, unique_filename)
    
    # If the string contains a data URI scheme prefix (like "data:image/png;base64,"),
    # remove it.
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",")[1]
    
    # Decode the base64 string to binary image data
    image_data = base64.b64decode(base64_string)
    
    # Write the image data to a file in binary mode
    with open(file_path, "wb") as image_file:
        image_file.write(image_data)
    
    return file_path

img_width = 200
img_height = 50

# Create the inference pipeline
predict_fn = create_inference_pipeline(model_path, img_width, img_height)

# Create asyncio events to signal when the request and response are captured
request_event = asyncio.Event()
response_event = asyncio.Event()

captured_request_id = None

async def outgoing_network_monitor(event):
    """
    Captures request events and stores the request ID when it finds the matching URL.
    """
    global captured_request_id
    req: uc.cdp.network.Request = event.request
    if '=NEcapchaId' in req.url:
        captured_request_id = event.request_id  # Store the request ID
        print(f"Captured Request ID: {captured_request_id}, URL: {req.url}")
        request_event.set()  # Signal that we've captured the request ID

async def response_received(event):
    """
    When a network response is received, this function matches it to the stored request,
    then attempts to process it. It waits for the response body and for specific page elements.
    """
    if event.request_id == captured_request_id:
        print(f"Matched response for Request ID: {event.request_id}")
        response_event.set()  # Signal that the request responded

async def main():
    id_ = "1001527983"
    phone = "543929796" 
    browser = await uc.start()
    page = await browser.get('https://www.absher.sa/wps/vanityurl/en/resetpasswordindividuals')
    await page.wait_for("img.captchaImg", timeout=60)

    # Add network event handlers
    page.add_handler(uc.cdp.network.RequestWillBeSent, outgoing_network_monitor)
    page.add_handler(uc.cdp.network.ResponseReceived, response_received)
    await page.wait(2)
    
    userid = await page.select("input.userNameId")
    mobile_number = await page.select("input.mobileNumberText")
    captcha_image = await page.select("img.captchaImg")
    captcha_input = await page.select("input.captchaNum")
    submit_button = await page.select("input.next")

    # Wait until the network event with the desired request ID is captured
    print("Waiting for the request ID to be captured...")
    await page.evaluate('captchaReload();', await_promise=False)
    await request_event.wait()
    print("Request ID captured:", captured_request_id)
    await response_event.wait()

    # Get the response body using the captured request ID
    body, _ = await page.send(uc.cdp.network.get_response_body(request_id=captured_request_id))
    print(f"Response Body: {body[:150]}...")
    captcha_path = save_image_from_base64(body)
    
    # Process captcha image prediction
    predicted_text = predict_fn(captcha_path)
    
    print(f"Predicted CAPTCHA: {predicted_text}")
    await captcha_input.send_keys(predicted_text)
    await userid.send_keys(id_)
    await mobile_number.send_keys(phone)
    await submit_button.click()

    await page.sleep(1)

    ## Start authentication check here ##
    # Wait for the header element that indicates the next page state
    header = await page.wait_for('h1[tabindex="0"]', timeout=60)
    # Attempt to find the warning message element, if present
    warning_text = ('Sorry, the registered number is not activated please visit Absher self-service kiosk device or through '
                    'the General Directorate of Passport or Civil affairs offices .')
    fail = await page.find_element_by_text(warning_text)

    # Extract header text (adjust method based on your API)
    header_text = await header.get_html()
    
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

    await page.wait(500)

if __name__ == '__main__':
    uc.loop().run_until_complete(main())
