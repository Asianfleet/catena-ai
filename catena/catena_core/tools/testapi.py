import requests

# API URL
url = "http://ea406b5d-babe-a247-9228-450f1af6f9aa.ofalias.com:24991/sdapi/v1/txt2img"

# JSON payload
payload = {
    "text": "A beautiful sunset over the mountains",
    "width": 512,
    "height": 512
}

# Send POST request
response = requests.post(url, json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response JSON
    result = response.json()
    
    # Assuming the image data is in base64 format under 'images' key
    import base64
    from PIL import Image
    from io import BytesIO
    
    # Decode the base64 image data
    image_data = base64.b64decode(result['images'][0])
    
    # Open the image using PIL
    image = Image.open(BytesIO(image_data))
    
    # Save the image to a file
    image.save("sunset.jpg")
    print("Image saved as sunset.jpg")
else:
    print(f"Failed to generate image. Status code: {response.status_code}")
    print("Response:", response.text)
