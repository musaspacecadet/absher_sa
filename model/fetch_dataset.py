import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Set up headers for the request
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0',
    'Accept': 'image/avif,image/webp,image/png,image/svg+xml,image/*;q=0.8,*/*;q=0.5',
    'Accept-Language': 'en-US,en;q=0.5',
    # 'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Connection': 'keep-alive',
    'Referer': 'https://www.absher.sa/wps/portal/individuals/static/resetpassword/!ut/p/z1/04_iUlDg4tKPAFJABjKBwtGPykssy0xPLMnMz0vM0Y_Qj4wyizd1DnD2tPA1NnQPCDU3MHIzN_FyNvN2DzMx1g_Hq8DAXD-KGP0GOICjAXH68SiIwm-8FyELglPzgJZE4VUG8iYhiyKBDjWPdw_w8HQ3cDfw8fcPcjYIDDYOcHVy9DQ0CjLXD9aPdNKPzCowCHZ11C_IDQ2NqHTNMi1wVAQAkvxxJg!!/dz/d5/L2dBISEvZ0FBIS9nQSEh/p0/IZ7_GPHIG0G0LOORC0QS3PEBAI12R7=CZ6_5CPCI8M31GPU702F74JC6KGV43=LA0=Ejavax.servlet.include.path_info!QCPuserValidation.jsp==/',
    # 'Cookie': 'JSESSIONID=0000JFjqmVYtyD-aSKrnAosaLIc:a4fg244b; TS01266e8c=016457516a910b40bd7f7c568aca193bb237e3e52e61bfa08d808a232fbd1ba331927e69fb0a2787c2287bbd20dfffbc8bbf5aad96; DigestTracker=AAABlNIt854; TS01921c89=016457516a910b40bd7f7c568aca193bb237e3e52e61bfa08d808a232fbd1ba331927e69fb0a2787c2287bbd20dfffbc8bbf5aad96; dtCookie=v_4_srv_1_sn_1DD76D2ABA5CBC9631E7081A0A09BA49_perc_100000_ol_0_mul_1_app-3A8d5b0733700157f3_1; cookie=!4K+H2XFiG6fg/ndtK9kxMPy8JkuOkkA+wPkjBKrf3cQni0eEnEL4NuRgRylWCash+DAcZl495oDmNQ==; TS0179031e=016457516a3554a70435e83a5bcc810a0a5c727a142eddee52a35511480bbbd732e1dcf32e0e99ad2661b0a43a48a78dc6f24d5753; TSd3408cbf027=08183185c1ab200067395ff7f3ef027f3928e9600d3f4bcf6f6a4c4ca51456d3e5f4eaf8dcea6e0b08317fa8f111300077a204cd8921d5593d77a34545af05dff412f5146a7be983dd5e97b404ef5440d4c78a59ca0c2ede7a042427d2100979; sessionId=0.09426895721544015; TS011185d8=016457516ab2071b1209ca72b769f8514e4822fb2c957894c7f0932c519a902831e4e75b8280db3cfd70622e109b3fc81feaa971b5; ADRUM=s=1738691955865&r=https%3A%2F%2Fwww.absher.sa%2Fwps%2Fportal%2Findividuals%2Fstatic%2Fresetpassword%2F!ut%2Fp%2Fz1%2F04_iUlDg4tKPAFJABjKBwtGPykssy0xPLMnMz0vM0Y_Qj4wyizd1DnD2tPA1NnQPCDU3MHIzN_FyNvN2DzMx1g_Hq8DAXD-KGP0GOICjAXH68SiIwm-8FyELglPzgJZE4VUG8iYhiwpyQ0MjKl0zATIw1qI!%2Fdz%2Fd5%2FL2dBISEvZ0FBIS9nQSEh%2Fp0%2FIZ7_GPHIG0G0LOORC0QS3PEBAI12R7%3DCZ6_5CPCI8M31GPU702F74JC6KGV43%3DLA0%3DEjavax.servlet.include.path_info!QCPinitialView.jsp%3D%3D%2F%3F961670456',
    'Sec-Fetch-Dest': 'image',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'same-origin',
    'Priority': 'u=5, i',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}

# Define the CAPTCHA endpoint
url = 'https://www.absher.sa/wps/portal/individuals/static/resetpassword/!ut/p/z1/04_iUlDg4tKPAFJABjKBwtGPykssy0xPLMnMz0vM0Y_Qj4wyizd1DnD2tPA1NnQPCDU3MHIzN_FyNvN2DzMx1g_Hq8DAXD-KGP0GOICjAXH68SiIwm-8FyELglPzgJZE4VUG8iYhiyKBDjWPdw_w8HQ3cDfw8fcPcjYIDDYOcHVy9DQ0CjLXD9aPdNaPzCowCHZ11C_IDQ2NqHTNMi1wVAQAUphZMQ!!/dz/d5/L2dBISEvZ0FBIS9nQSEh/p0/IZ7_GPHIG0G0LOORC0QS3PEBAI12R7=CZ6_5CPCI8M31GPU702F74JC6KGV43=NEcapchaId!0.7270301247687265==/1738692337733'


# Directory to save CAPTCHA images
output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

# Number of images to download - You can change this to increase dataset size
num_images = 5000

# Max threads to use
max_threads = 20

def download_captcha(image_id):
    """Download a single CAPTCHA image."""
    image_path = os.path.join(output_dir, f'captcha_{image_id:04d}.jpeg')

    # Check if the image already exists
    if os.path.exists(image_path):
        return 'exists'  # Indicate that the image already exists

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Save image to file
            with open(image_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download image {image_id}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading image {image_id}: {e}")
        return False

# Find the highest existing image ID to determine the starting point
existing_images = [f for f in os.listdir(output_dir) if f.startswith('captcha_') and f.endswith('.png')]
if existing_images:
    highest_id = max([int(f.split('_')[1].split('.')[0]) for f in existing_images])
    start_id = highest_id + 1
else:
    start_id = 1

# Determine the actual number of images to download
images_to_download = num_images - (start_id - 1)

if images_to_download <= 0:
    print("Desired number of images already downloaded.")
else:
    # Multithreaded downloading
    with ThreadPoolExecutor(max_threads) as executor:
        futures = {executor.submit(download_captcha, start_id + i): start_id + i for i in range(images_to_download)}

        # Progress bar to track completed tasks
        for future in tqdm(as_completed(futures), total=images_to_download, desc="Downloading CAPTCHA images"):
            result = future.result()
            if result == 'exists':
                # print(f"Image {futures[future]:04d} already exists.")
                pass

    print(f"Downloaded new CAPTCHA images to '{output_dir}'. Total images: {num_images}")