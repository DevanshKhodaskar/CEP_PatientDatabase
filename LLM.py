
import base64
import json
from groq import Groq
import dotenv
import os
from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client (Replace with your actual API key)
client = Groq(api_key=api_key)  # Replace with your valid API key

# Read image and convert to Base64
image_path = "sample certificate.jpg"  # Ensure this file exists in the same directory
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Define request payload with Base64 encoded image
completion = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract structured data from this form and return it in JSON format with fields: "
                            "{ 'name': '', 'last_name': '', 'gender': '', 'dob': '', 'contact_no': '', "
                            "'blood_group': '', 'medical_details': {} }"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.5,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    response_format={"type": "json_object"},
    stop=None,
)

# Extract the JSON response correctly
structured_data = json.loads(completion.choices[0].message.content)

# Print formatted JSON output
print(json.dumps(structured_data, indent=4))

# Save JSON output to a file
with open("output.json", "w") as f:
    json.dump(structured_data, f, indent=4)
