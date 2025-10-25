import csv
import requests
import time
import os
from dotenv import load_dotenv  # Import the function

# --- Configuration ---

# The CSV file to read prompts from.
# It MUST have a column named 'prompt'.
# Example:
# id,prompt,category
# 1,"What is the capital of France?","geography"
# 2,"Summarize the plot of 'Hamlet'.","literature"
INPUT_CSV_FILE = 'prompts.csv'

# The CSV file to write results to.
OUTPUT_CSV_FILE = 'responses.csv'

# The name of the column in your input CSV that contains the prompts.
PROMPT_COLUMN_NAME = 'prompt'

# The Gemini model to use.
MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'

# API URL template.
API_URL_TEMPLATE = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={{api_key}}'

# Delay between *different* API requests in seconds.
DELAY_BETWEEN_REQUESTS = 1

# --- New Retry Configuration ---
# Max number of retries for a single prompt if it fails.
MAX_RETRIES = 5
# Initial time to wait (in seconds) before the first retry.
# This will double after each failed attempt (e.g., 5s, 10s, 20s...).
INITIAL_BACKOFF_SEC = 5
# --- End Configuration ---


def get_gemini_response(prompt: str, api_key: str) -> str:
    """
    Sends a prompt to the Gemini API and returns the text response.
    Implements retry logic with exponential backoff for server errors.

    Args:
        prompt: The text prompt to send to the model.
        api_key: Your Google AI Studio API key.

    Returns:
        The generated text response or an error message.
    """
    api_url = API_URL_TEMPLATE.format(api_key=api_key)
    headers = {'Content-Type': 'application/json'}
    payload = {
        'contents': [{
            'parts': [{'text': prompt}]
        }]
    }

    current_delay = INITIAL_BACKOFF_SEC
    
    for attempt in range(MAX_RETRIES):
        # We need a variable to hold the response text in case of JSON errors
        response_text = ""
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response_text = response.text # Store raw text in case .json() fails
            
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            result = response.json()
            
            # Extract the text from the response
            if 'candidates' in result and result['candidates']:
                if 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                    # Success! Return the response.
                    return result['candidates'][0]['content']['parts'][0]['text']
            
            # If the expected structure isn't found
            return f"ERROR: Unexpected JSON response structure: {result}"

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            # Retry on: 429 (Too Many Requests), 500 (Internal Error), 503 (Overloaded)
            retryable_codes = [429, 500, 503, 504]

            if status_code in retryable_codes and (attempt + 1) < MAX_RETRIES:
                print(f"  WARNING: HTTP {status_code}. Attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff
                continue # Go to the next iteration of the loop
            else:
                # Final attempt failed or it's a non-retryable error (like 400, 401, 404)
                return f"ERROR: HTTP error occurred: {http_err} - {response_text}"
        
        except requests.exceptions.RequestException as req_err:
            # This catches network errors (e.g., DNS failure, connection refused)
            if (attempt + 1) < MAX_RETRIES:
                print(f"  WARNING: Request exception. Attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {current_delay}s... ({req_err})")
                time.sleep(current_delay)
                current_delay *= 2
                continue # Go to the next iteration of the loop
            else:
                return f"ERROR: Request exception occurred after {MAX_RETRIES} attempts: {req_err}"
        
        except (KeyError, IndexError, TypeError, requests.exceptions.JSONDecodeError) as json_err:
            # This means the response was likely not valid JSON or parsing failed.
            # Retrying won't help.
            return f"ERROR: Failed to parse JSON response: {json_err} - Response: {response_text}"
        
        except Exception as e:
            # Catch-all for any other unexpected error
            return f"ERROR: An unexpected error occurred: {e}"

    # If the loop completes without returning, all retries failed.
    return f"ERROR: Failed to get response for prompt after {MAX_RETRIES} attempts."


def process_prompts_from_csv():
    """
    Reads prompts from the input CSV, gets responses, and writes to the output CSV.
    """
    print("Starting batch prompt processing...")

    # 1. Load environment variables from a .env file (if it exists)
    load_dotenv()

    # 2. Get API Key
    # This will now first check the .env file, then other environment variables
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print(f"Error: GEMINI_API_KEY environment variable not set.")
        print("Please set your API key in a .env file or as an environment variable before running.")
        return

    print(f"Reading prompts from '{INPUT_CSV_FILE}'...")
    
    try:
        # 3. Read input CSV and prepare for output
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            
            input_fieldnames = reader.fieldnames
            if not input_fieldnames:
                print(f"Error: The input CSV '{INPUT_CSV_FILE}' is empty or not valid.")
                return

            if PROMPT_COLUMN_NAME not in input_fieldnames:
                print(f"Error: Input CSV must have a column named '{PROMPT_COLUMN_NAME}'.")
                print(f"Found columns: {', '.join(input_fieldnames)}")
                return
            
            # All original columns + the new 'response' column
            output_fieldnames = input_fieldnames + ['response']
            
            rows = list(reader) # Read all rows into memory

    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_FILE}'")
        print("Please create this file and add your prompts.")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    print(f"Found {len(rows)} prompts to process. Writing results to '{OUTPUT_CSV_FILE}'...")

    # 4. Process each row and write to output CSV
    try:
        with open(OUTPUT_CSV_FILE, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()

            for i, row in enumerate(rows):
                prompt = row.get(PROMPT_COLUMN_NAME)

                if not prompt:
                    print(f"Skipping row {i+2} (0-indexed + header): Prompt is empty.")
                    response = "ERROR: Prompt was empty"
                else:
                    print(f"Processing row {i+2}/{len(rows)+1}: '{prompt[:70]}...'")
                    response = get_gemini_response(prompt, api_key)
                    
                    # Add a delay to be respectful of API rate limits
                    time.sleep(DELAY_BETWEEN_REQUESTS)

                # Create the new row for the output file
                output_row = row.copy()
                output_row['response'] = response
                writer.writerow(output_row)

    except IOError as e:
        print(f"Error writing to output file '{OUTPUT_CSV_FILE}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

    print("\nBatch processing complete.")
    print(f"Results saved to '{OUTPUT_CSV_FILE}'.")


if __name__ == "__main__":
    # To run this script:
    # 1. Install the required 'python-dotenv' library:
    #    pip install python-dotenv requests
    #    (Added 'requests' just in case it wasn't installed)
    #
    # 2. Save this file as 'batch_processor.py'.
    # 3. Create a 'prompts.csv' file in the same directory.
    #    Example 'prompts.csv' content:
    #    id,prompt
    #    1,"What is the capital of France?"
    #    2,"Who wrote '1984'?"
    #
    # 4. Get your API key from Google AI Studio.
    # 5. Create a file named '.env' in the same directory.
    #    Inside the .env file, add this line:
    #    GEMINI_API_KEY='YOUR_API_KEY_HERE'
    #
    # 6. Run the script from your terminal:
    #    python batch_processor.py
    
    process_prompts_from_csv()