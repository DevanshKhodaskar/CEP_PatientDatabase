import base64
import json
from groq import Groq
import dotenv
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)

# Read image and convert to Base64
image_path = "DataFilledHumanHandWritting2.png"  # Path to the cheque image
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

prompt_text = """
Sure! Here's the revised **prompt** with the hospital name removed, maintaining clarity, structure, and strong prompt engineering principles:

Prompt:

You are an intelligent and detail-oriented assistant specialized in document analysis and data extraction. Your task is to analyze an image of a **patient's treatment card** and extract structured information using Optical Character Recognition (OCR).

Task Objectives:

1. **Perform OCR**: Accurately read and interpret printed text from the image.
2. **Extract and Normalize Data**: Identify and extract specific information from the card while correcting common spelling errors (e.g., treat “pervious” as “previous”).
3. **Handle Missing or Illegible Data**: If a field is missing, obscured, or illegible, return its value as `null`.
4. **Format the Output**: Return the result as a clean, valid **JSON object only**, with no extra explanations or text.


Required Output Fields:

Return the extracted data with the following keys:

* `book_number`
* `registration_number`
* `name`
* `age`
* `date_of_birth`
* `sex`
* `caste`
* `mobile_number`
* `aadhar_number`
* `address`
* `date_of_admission`
* `mother's_name`
* `relatives` *(combine multiple lines into one string)*
* `blood_group`
* `leprosy_type` *(either `"MB"` or `"PB"` only)*
* `mdt_status` *(must be `"Cured"`, `"Under MDT"`, or `"Unknown"`)*
* `deformity_status`
* `duration_of_disease`
* `previous_occupation`



Special Instructions:

* Combine multi-line fields like `address` or `relatives` into a single-line string.
* Normalize spelling inconsistencies (e.g., “pervious” = “previous”).
* Ensure the final output is a clean and correctly structured JSON—**no headings, labels, or additional commentary.**



Output Format Example:

json
{
  "book_number": "1234",
  "registration_number": "5678",
  "name": "John Doe",
  "age": "45",
  "date_of_birth": "1978-05-12",
  "sex": "Male",
  "caste": "OBC",
  "mobile_number": "9876543210",
  "aadhar_number": "123412341234",
  "address": "123 Main Street, City Name, State",
  "date_of_admission": "2023-03-14",
  "mother's_name": "Mary Doe",
  "relatives": "Father: Robert Doe, Brother: James Doe",
  "blood_group": "B+",
  "leprosy_type": "MB",
  "mdt_status": "Under MDT",
  "deformity_status": "Grade II",
  "duration_of_disease": "2 years",
  "previous_occupation": "Farmer"
}

Begin the extraction now and return only the structured JSON response.

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
