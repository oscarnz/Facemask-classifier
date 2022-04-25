import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

def process(image, server_url: str):
    # handle and read the images 
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    r = []

    #send input given by the user to fastAPI backend using Post method 
    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)
    
    # convert into dictionary
    rdict = r.json()

    return str(rdict)

def load_image(image_file):
    # opens the image file
	img = Image.open(image_file)
    
	return img