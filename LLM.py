import base64
import json
from groq import Groq
import dotenv
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)

# Read image and convert to Base64
image_path = "DataFilledHumanHandWritting2.png"  # Path to the cheque image
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Define the prompt
prompt_text = """
You are an intelligent assistant tasked with extracting structured information from a patient's treatment card image issued by SITA RATAN LEPROSY HOSPITAL, ANANDWAN.

1. Use OCR to read printed text from the image.
2. Extract the data into JSON format with the following keys:
- book_number
- registration_number
- name
- age
- date_of_birth
- sex
- caste
- mobile_number
- aadhar_number
- address
- date_of_admission
- mother's_name
- relatives
- blood_group
- leprosy_type (either "MB" or "PB")
- mdt_status (either "Cured", "Under MDT", or "Unknown")
- deformity_status
- duration_of_disease
- previous_occupation

3. If any field is empty or not visible in the image, assign its value as `null`.
4. For multi-line fields like address or relatives, combine the lines into a single string.
5. Return the final response in a clean JSON format only, with no extra commentary.

Handle spelling inconsistencies (e.g., “pervious” should be treated as “previous”) and include them correctly in the JSON output.

"""

# Create request
completion = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
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
