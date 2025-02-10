import logging
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from botasaurus.browser import browser, Driver
from botasaurus.browser import Driver, cdp
from botasaurus_driver.core import util
from botasaurus_driver.core.env import is_docker
from botasaurus_driver.core.browser import Browser, terminate_process, wait_for_graceful_close, delete_profile
from model import create_inference_pipeline, model_path
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_system.log'),
        logging.StreamHandler()
    ]
)

# Configuration
MAX_WORKERS = 3
RETRY_LIMIT = 3
COOLDOWN_PERIOD = 2
PAGE_LOAD_TIMEOUT = 30
ELEMENT_WAIT_TIMEOUT = 10

class FormState(Enum):
    INIT = "initializing"
    PAGE_LOADING = "loading_page"
    PAGE_LOADED = "page_loaded"
    FILLING_FORM = "filling_form"
    SOLVING_CAPTCHA = "solving_captcha"
    SUBMITTING = "submitting"
    WAITING_RESPONSE = "waiting_response"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    RETRY = "retry"

@dataclass
class UserRecord:
    id: str
    phone: str
    state: FormState = FormState.INIT
    attempts: int = 0
    result: Optional[str] = None
    error_message: Optional[str] = None
    last_captcha: Optional[str] = None
    form_data: Dict[str, Any] = None

class FormStateMachine:
    def __init__(self):
        self.predict_fn = create_inference_pipeline(model_path, 200, 50)
        self.state_lock = threading.Lock()
        self.results_queue = Queue()
        
    def transition_state(self, user: UserRecord, new_state: FormState, message: str = None):
        """Thread-safe state transition with logging"""
        with self.state_lock:
            old_state = user.state
            user.state = new_state
            logging.info(f"User {user.id}: {old_state.value} -> {new_state.value} {f': {message}' if message else ''}")

    def process_user(self, user: UserRecord) -> UserRecord:
        """Process a single user through the form state machine"""
        logger = logging.getLogger(f"User_{user.id}")
        driver = None
        
        try:
            while user.attempts < RETRY_LIMIT and user.state not in [FormState.SUCCESS, FormState.FAILED]:
                user.attempts += 1
                logger.info(f"Starting attempt {user.attempts}/{RETRY_LIMIT}")
                
                try:
                    if not driver:
                        driver = Driver(block_images_and_css=True, profile=False)
                    
                    self.handle_form_submission(driver, user, logger)
                    
                except Exception as e:
                    error_msg = f"Error during attempt {user.attempts}: {str(e)}"
                    logger.error(error_msg)
                    user.error_message = error_msg
                    
                    if user.attempts >= RETRY_LIMIT:
                        self.transition_state(user, FormState.ERROR, error_msg)
                    else:
                        self.transition_state(user, FormState.RETRY, "Will retry after cooldown")
                        time.sleep(COOLDOWN_PERIOD)
                        
                        # Clear cookies and reload page for retry
                        if driver:
                            driver.delete_cookies_and_local_storage()
                            
        finally:
            if driver:
                self.close(driver._browser)
                logger.info("Browser session closed")
            
            self.results_queue.put(user)
            logger.info(f"Final state: {user.state.value}")
            
        return user

    def handle_form_submission(self, driver: Driver, user: UserRecord, logger: logging.Logger):
        """Handle the form submission process through various states"""
        
        # State: Page Loading
        self.transition_state(user, FormState.PAGE_LOADING)
        page = driver.get("https://www.absher.sa/wps/vanityurl/en/resetpasswordindividuals")
        
        # State: Page Loaded - Get Form Elements
        self.transition_state(user, FormState.PAGE_LOADED)
        form_elements = self.get_form_elements(page)
        if not form_elements:
            raise Exception("Failed to locate form elements")
        
        # State: Solving CAPTCHA
        self.transition_state(user, FormState.SOLVING_CAPTCHA)
        captcha_result = self.solve_captcha(form_elements['captcha_image'])
        user.last_captcha = captcha_result
        logger.info(f"CAPTCHA solution attempt: {captcha_result}")
        
        # State: Filling Form
        self.transition_state(user, FormState.FILLING_FORM)
        self.fill_form(form_elements, user, captcha_result)
        
        # State: Submitting
        self.transition_state(user, FormState.SUBMITTING)
        form_elements['submit_button'].click()
        
        # State: Waiting Response
        self.transition_state(user, FormState.WAITING_RESPONSE)
        result = self.check_submission_result(page)
        
        # Update final state based on result
        if result["success"]:
            self.transition_state(user, FormState.SUCCESS, result["message"])
            user.result = result["message"]
        else:
            if user.attempts >= RETRY_LIMIT:
                self.transition_state(user, FormState.FAILED, result["message"])
            else:
                self.transition_state(user, FormState.RETRY, result["message"])
            user.result = result["message"]

    def get_form_elements(self, page):
        """Get all form elements with proper error handling"""
        try:
            return {
                'captcha_image': page.select("img.captchaImg"),
                'captcha_input': page.select("input.captchaNum"),
                'userid': page.select("input.userNameId"),
                'mobile_number': page.select("input.mobileNumberText"),
                'submit_button': page.select("input.next")
            }
        except Exception as e:
            raise Exception(f"Failed to locate form elements: {str(e)}")



    def solve_captcha(self, captcha_image) -> str:
        """Handle CAPTCHA solving with error checking, using a temporary file"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image_path = temp_file.name
            captcha_image.save_screenshot(image_path)
        
        try:
            predicted_text = self.predict_fn(image_path)
            if not predicted_text:
                raise Exception("CAPTCHA prediction failed")
            return predicted_text
        finally:
            os.remove(image_path)


    def fill_form(self, elements, user: UserRecord, captcha_text: str):
        """Fill form with error handling"""
        try:
            elements['userid'].send_keys(user.id)
            elements['mobile_number'].send_keys(user.phone)
            elements['captcha_input'].send_keys(captcha_text)
            user.form_data = {
                'id': user.id,
                'phone': user.phone,
                'captcha': captcha_text
            }
        except Exception as e:
            raise Exception(f"Failed to fill form: {str(e)}")

    def check_submission_result(self, page) -> dict:
        """Check form submission result with detailed error handling"""
        try:
            header = page.wait_for('h1[tabindex="0"]', timeout=ELEMENT_WAIT_TIMEOUT)
            warning_text = ('Sorry, the registered number is not activated please visit Absher self-service kiosk device or through '
                          'the General Directorate of Passport or Civil affairs offices .')
            fail = page.find_element_by_text(warning_text)
            
            header_text = header.get_html()
            
            if "Security Validation" in header_text and not fail:
                return {
                    "success": True,
                    "message": "Authentication Successful: Security validation passed"
                }
            elif fail:
                return {
                    "success": False,
                    "message": f"Authentication Failed: {warning_text}"
                }
            else:
                return {
                    "success": False,
                    "message": f"Authentication Failed: Unexpected header text: {header_text}"
                }
        except Exception as e:
            raise Exception(f"Failed to check submission result: {str(e)}")
    
    def close(self, browser: Browser):
        # close gracefully
        if browser.connection:
            browser.connection.send(cdp.browser.close())
        
        browser.close_tab_connections()
        browser.close_browser_connection()

    
        if browser._process:
            if not wait_for_graceful_close(browser._process):
                terminate_process(browser._process)
        browser._process = None
        browser._process_pid = None

        if browser.config.is_temporary_profile:
            delete_profile(browser.config.profile_directory)
        browser.config.close()
        instances = util.get_registered_instances()
        try:
            instances.remove(browser)
        except KeyError:
            pass

        if is_docker:
            util.close_zombie_processes()

class UserValidationSystem:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.state_machine = FormStateMachine()
        self.output_file = input_file.replace('.xlsx', '_results.xlsx')
        
    def load_users(self) -> list[UserRecord]:
        """Load user records from Excel file with error handling"""
        try:
            df = pd.read_excel(self.input_file)
            required_columns = {'id', 'mobile'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Excel file must contain columns: {required_columns}")
            
            return [UserRecord(str(row['id']), str(row['mobile'])) 
                   for _, row in df.iterrows()]
        except Exception as e:
            logging.error(f"Failed to load users from {self.input_file}: {str(e)}")
            raise
    
    def save_results(self):
        """Save results to Excel with error handling"""
        try:
            results = []
            while not self.state_machine.results_queue.empty():
                user = self.state_machine.results_queue.get()
                results.append({
                    'id': user.id,
                    'phone': user.phone,
                    'state': user.state.value,
                    'attempts': user.attempts,
                    'result': user.result,
                    'error_message': user.error_message,
                    'last_captcha': user.last_captcha
                })
            
            df = pd.DataFrame(results)
            df.to_excel(self.output_file, index=False)
            logging.info(f"Results saved to {self.output_file}")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            raise
    
    def run(self):
        """Run the validation system with improved error handling"""
        try:
            users = self.load_users()
            logging.info(f"Starting validation for {len(users)} users")
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                executor.map(self.state_machine.process_user, users)
            
            self.save_results()
            logging.info("Validation process completed")
            
        except Exception as e:
            logging.error(f"System error: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        system = UserValidationSystem("demo.xlsx")
        system.run()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")