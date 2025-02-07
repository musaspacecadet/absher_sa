import asyncio
import logging
import os
import uuid
import base64
import pandas as pd
from datetime import datetime
from typing import List, Dict

import zendriver as uc
from model import create_inference_pipeline, model_path

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def save_image_from_base64(base64_string, output_dir="images"):
    """Save a base64-encoded image to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    unique_filename = f"{uuid.uuid4()}.png"
    file_path = os.path.join(output_dir, unique_filename)
    
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",")[1]
    
    try:
        image_data = base64.b64decode(base64_string)
        with open(file_path, "wb") as image_file:
            image_file.write(image_data)
        logging.info(f"Saved captcha image to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save captcha image: {e}")
        raise e
    return file_path

interactive_script = """
(function waitForInteractive() {
    return new Promise(function(resolve, reject) {
        var pollInterval = 100; // Check every 100ms
        var maxAttempts = 10000 / pollInterval; // Timeout after 10 seconds
        var attempts = 0;
        var interval = setInterval(function() {
            var state = document.readyState;
            if (state === "complete") {
                clearInterval(interval);
                console.log("Page is interactive");
                resolve("Page is interactive");
            }
            attempts++;
            if (attempts >= maxAttempts) {
                clearInterval(interval);
                console.log("Timeout waiting for page to become interactive");
                reject("Timeout waiting for page to become interactive");
            }
        }, pollInterval);
    });
})()
"""

class TabProcessor:
    def __init__(self, browser, predict_fn):
        self.browser: uc.Browser = browser
        self.predict_fn = predict_fn
        self.request_event = asyncio.Event()
        self.response_event = asyncio.Event()
        self.captured_request_id = None
        self.max_captcha_attempts = 3  # Maximum number of CAPTCHA attempts

    async def outgoing_network_monitor(self, event):
        try:
            req: uc.cdp.network.Request = event.request
            if '=NEcapchaId' in req.url:
                self.captured_request_id = event.request_id
                self.request_event.set()
                logging.debug(f"Captured request id: {self.captured_request_id}")
        except Exception as e:
            logging.error(f"Error in outgoing_network_monitor: {e}")

    async def response_received(self, event):
        try:
            if event.request_id == self.captured_request_id:
                self.response_event.set()
                logging.debug(f"Response received for request id: {self.captured_request_id}")
        except Exception as e:
            logging.error(f"Error in response_received: {e}")

    async def try_captcha(self, page: uc.Tab, user_id, phone):
        await page.evaluate(interactive_script, await_promise=False)
        page.feed_cdp(uc.cdp.dom.enable())

        # Reset events for new request
        self.request_event.clear()
        self.response_event.clear()
        self.captured_request_id = None

        try:
            # Get input fields
            captcha_input = await page.wait_for("input.captchaNum", timeout=200)
            userid_field = await page.wait_for("input.userNameId", timeout=200)
            mobile_number_field = await page.wait_for("input.mobileNumberText", timeout=200)
        except Exception as e:
            logging.error(f"Error waiting for input fields: {e}")
            return False

        try:
            # Clear any existing inputs
            await captcha_input.clear_input()
            await userid_field.clear_input()
            await mobile_number_field.clear_input()
        except Exception as e:
            logging.error(f"Error clearing input fields: {e}")
            return False

        try:
            # Trigger captcha reload and wait for network events
            await page.evaluate('captchaReload();', await_promise=False)
            logging.info("Triggered captcha reload")
            await asyncio.wait_for(self.request_event.wait(), timeout=100)
            await asyncio.wait_for(self.response_event.wait(), timeout=100)
        except asyncio.TimeoutError:
            logging.error("Timeout waiting for network events after captcha reload")
            return False
        except Exception as e:
            logging.error(f"Error during captcha reload network event wait: {e}")
            return False

        try:
            # Get captcha image and process it
            body, _ = await page.send(uc.cdp.network.get_response_body(request_id=self.captured_request_id))
            captcha_path = save_image_from_base64(body)
            predicted_text = self.predict_fn(captcha_path)
            logging.info(f"Predicted captcha text: {predicted_text}")
        except Exception as e:
            logging.error(f"Error processing captcha image: {e}")
            return False

        try:
            # Fill in the captcha and other fields
            await captcha_input.send_keys(predicted_text)
            await userid_field.send_keys(str(user_id))
            await mobile_number_field.send_keys(str(phone))
            logging.info("Filled in form fields with user credentials and captcha")
        except Exception as e:
            logging.error(f"Error filling in form fields: {e}")
            return False

        try:
            submit_button = await page.select("input.next")
            await submit_button.click()
            logging.info("Clicked submit button")
        except Exception as e:
            logging.error(f"Error clicking submit button: {e}")
            return False

        try:
            # Wait briefly for any error message to appear
            await page.wait(2)
            await page.evaluate(interactive_script, await_promise=False)
            page.feed_cdp(uc.cdp.dom.enable())
        except Exception as e:
            logging.warning(f"Wait after submit encountered an issue: {e}")

        try:
            error_msg = await page.find_element_by_text("Entered image code is not correct please try again", best_match=True)
            if error_msg:
                logging.info(f"Incorrect CAPTCHA attempt: {predicted_text}")
                return False
            else:
                return True
        except Exception as e:
            # If element not found or any other exception, assume CAPTCHA is correct.
            logging.info("No error message detected, assuming captcha success")
            return True

    async def process_user(self, user_id: int, phone: str) -> Dict:
        page = None
        try:
            logging.info(f"Processing user {user_id} with phone {phone}")
            page = await self.browser.get('https://www.absher.sa/wps/vanityurl/en/resetpasswordindividuals', new_window=True)
            
            await page.wait_for("img.captchaImg", timeout=60)
            await page.evaluate(interactive_script, await_promise=False)
            page.feed_cdp(uc.cdp.dom.enable())
            logging.info("Captcha image loaded on page")

            # Add event handlers for monitoring captcha reload network events
            page.add_handler(uc.cdp.network.RequestWillBeSent, self.outgoing_network_monitor)
            page.add_handler(uc.cdp.network.ResponseReceived, self.response_received)
            
            # Try CAPTCHA multiple times if needed
            captcha_success = False
            attempts = 0

            while not captcha_success and attempts < self.max_captcha_attempts:
                attempts += 1
                logging.info(f"CAPTCHA attempt {attempts} for user {user_id}")
                captcha_success = await self.try_captcha(page, user_id, phone)
                if not captcha_success:
                    logging.info(f"CAPTCHA attempt {attempts} failed for user {user_id}")
                    # Optionally, add recovery steps here (e.g., page refresh) before retrying
            if not captcha_success:
                logging.error(f"CAPTCHA attempts exhausted for user {user_id}")
                return {
                    "id": user_id,
                    "mobile": str(phone),
                    "result": "Failed: CAPTCHA attempts exhausted",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            try:
                await page.evaluate(interactive_script, await_promise=False)
                page.feed_cdp(uc.cdp.dom.enable())
                header = await page.wait_for('h1[tabindex="0"]', timeout=60)
                header_text = await header.get_html()
                logging.info(f"Header text received: {header_text}")
            except Exception as e:
                logging.error(f"Error waiting for header: {e}")
                return {
                    "id": user_id,
                    "mobile": str(phone),
                    "result": f"Error: {str(e)}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

            warning_text = ('Sorry, the registered number is not activated please visit Absher self-service kiosk device or through '
                            'the General Directorate of Passport or Civil affairs offices .')
            try:
                fail = await page.find_element_by_text(warning_text, best_match=True)
            except Exception:
                fail = None

            result = "Success" if "Security Validation" in header_text and not fail else "Failed"
            logging.info(f"User {user_id} processing result: {result}")
            return {
                "id": user_id,
                "mobile": str(phone),
                "result": result,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            logging.error(f"Exception processing user {user_id}: {e}")
            return {
                "id": user_id,
                "mobile": str(phone),
                "result": f"Error: {str(e)}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        finally:
            if page:
                try:
                    await page.close()
                    logging.debug(f"Page closed for user {user_id}")
                except Exception as e:
                    logging.error(f"Error closing page for user {user_id}: {e}")

async def process_batch(batch: pd.DataFrame, browser: uc.Browser, predict_fn, max_tabs: int = 3) -> List[Dict]:
    processors = [TabProcessor(browser, predict_fn) for _ in range(max_tabs)]
    results = []
    
    for i in range(0, len(batch), max_tabs):
        current_batch = batch.iloc[i:i + max_tabs]
        tasks = []
        
        for (idx, row), processor in zip(current_batch.iterrows(), processors):
            task = asyncio.create_task(processor.process_user(int(row['id']), str(row['mobile'])))
            tasks.append(task)
        
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error processing batch starting at index {i}: {e}")
            batch_results = []
        results.extend(batch_results)
        # Pause between batches to help with stability
        await asyncio.sleep(1)
    
    return results

async def main():
    try:
        # Load the Excel file
        df = pd.read_excel('demo.xlsx')
        df['id'] = df['id'].astype(int)
        df['mobile'] = df['mobile'].astype(str)
        logging.info("Excel file loaded successfully")

        # Drop duplicate rows (processing each unique user only once)
        unique_df = df.drop_duplicates(subset=["id", "mobile"]).reset_index(drop=True)
        logging.info(f"Processing {len(unique_df)} unique user(s) out of {len(df)} rows")

        # Initialize the CAPTCHA prediction model
        img_width = 200
        img_height = 50
        predict_fn = create_inference_pipeline(model_path, img_width, img_height)
        logging.info("Captcha prediction model initialized")

        # Start the browser
        browser = await uc.start()
        logging.info("Browser started")

        # Process all unique users
        results = await process_batch(unique_df, browser, predict_fn, max_tabs=3)
        logging.info("Batch processing completed")

        results_df = pd.DataFrame(results)
        results_df['id'] = results_df['id'].astype(int)
        results_df['mobile'] = results_df['mobile'].astype(str)
        # Merge results with the original data (this will repeat the result for duplicate rows)
        final_df = df.merge(results_df, on=["id", "mobile"], how='left')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'results_{timestamp}.xlsx'
        final_df.to_excel(output_file, index=False)
        logging.info(f"Results saved to {output_file}")

        await browser.stop()
        logging.info("Browser stopped")
    except Exception as e:
        logging.error(f"Error in main: {e}")


if __name__ == '__main__':
    uc.loop().run_until_complete(main())
