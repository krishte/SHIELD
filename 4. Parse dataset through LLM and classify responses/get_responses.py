import csv
import requests
import time
import os
import concurrent.futures  # <<< IMPORT FOR PARALLELISM
from dotenv import load_dotenv
from tqdm import tqdm

# --- Configuration ---

# The CSV file to read prompts from.
# It MUST have a column named 'prompt'.
INPUT_CSV_FILE = 'my_labeled_dataset.csv'

# The CSV file to write results to.
OUTPUT_CSV_FILE = 'dataset_with_responses.csv'

# The name of the column in your input CSV that contains the prompts.
PROMPT_COLUMN_NAME = 'prompt'

# The Gemini model to use.
MODEL_NAME = 'gemini-2.5-flash-lite'

# API URL template.
API_URL_TEMPLATE = f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={{api_key}}'

# --- Parallelism Configuration ---
# Number of parallel threads to run.
# This is the most important setting for speed.
# A good starting point is 5-10.
# WARNING: Setting this too high (e.g., 50) WILL get you rate-limited (HTTP 429).
# The script's retry logic will handle this, but it may not be faster.
# Adjust based on your API quota (e.g., Gemini Flash default is 60 requests/minute).
MAX_WORKERS = 10 
# --- End Configuration ---


# --- Retry Configuration ---
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 5
# --- End Configuration ---


def get_gemini_response(prompt: str, api_key: str) -> str:
    """
    Sends a prompt to the Gemini API and returns the text response.
    Implements retry logic with exponential backoff for server errors.
    (This function is now thread-safe and called by multiple workers)
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
        response_text = ""
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response_text = response.text 
            
            response.raise_for_status()

            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                if 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                    return result['candidates'][0]['content']['parts'][0]['text']
            
            return f"ERROR: Unexpected JSON response structure: {result}"

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            retryable_codes = [429, 500, 503, 504]

            if status_code in retryable_codes and (attempt + 1) < MAX_RETRIES:
                # Use tqdm.write to print thread-safe messages without breaking the bar
                tqdm.write(f"  WARNING (Prompt: '{prompt[:30]}...'): HTTP {status_code}. Attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= 2
                continue
            else:
                return f"ERROR: HTTP error occurred: {http_err} - {response_text}"
        
        except requests.exceptions.RequestException as req_err:
            if (attempt + 1) < MAX_RETRIES:
                tqdm.write(f"  WARNING (Prompt: '{prompt[:30]}...'): Request exception. Attempt {attempt + 1}/{MAX_RETRIES}. Retrying in {current_delay}s... ({req_err})")
                time.sleep(current_delay)
                current_delay *= 2
                continue
            else:
                return f"ERROR: Request exception occurred after {MAX_RETRIES} attempts: {req_err}"
        
        except (KeyError, IndexError, TypeError, requests.exceptions.JSONDecodeError) as json_err:
            return f"ERROR: Failed to parse JSON response: {json_err} - Response: {response_text}"
        
        except Exception as e:
            return f"ERROR: An unexpected error occurred: {e}"

    return f"ERROR: Failed to get response for prompt '{prompt[:30]}...' after {MAX_RETRIES} attempts."


def process_prompts_from_csv():
    """
    Reads prompts from the input CSV, gets responses in parallel, 
    and writes to the output CSV.
    """
    print("Starting batch prompt processing...")

    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print(f"Error: GEMINI_API_KEY environment variable not set.")
        return

    print(f"Reading prompts from '{INPUT_CSV_FILE}'...")
    
    try:
        with open(INPUT_CSV_FILE, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            
            input_fieldnames = reader.fieldnames
            if not input_fieldnames:
                print(f"Error: The input CSV '{INPUT_CSV_FILE}' is empty or not valid.")
                return

            if PROMPT_COLUMN_NAME not in input_fieldnames:
                print(f"Error: Input CSV must have a column named '{PROMPT_COLUMN_NAME}'.")
                return
            
            output_fieldnames = input_fieldnames + ['response']
            rows = list(reader) # Read all rows into memory

    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_CSV_FILE}'")
        return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    print(f"Found {len(rows)} prompts to process. Writing results to '{OUTPUT_CSV_FILE}'...")
    print(f"Using up to {MAX_WORKERS} parallel workers.")

    try:
        with open(OUTPUT_CSV_FILE, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
            writer.writeheader()

            # --- PARALLEL PROCESSING BLOCK ---
            
            # This dictionary will map a "future" (a running job) back to its original row data
            future_to_row = {}
            
            # We use a ThreadPoolExecutor to manage our worker threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                
                # First, submit all jobs to the executor
                for row in rows:
                    prompt = row.get(PROMPT_COLUMN_NAME)
                    
                    if not prompt:
                        # If the prompt is empty, don't submit it.
                        # We'll write it directly as an error.
                        output_row = row.copy()
                        output_row['response'] = "ERROR: Prompt was empty"
                        writer.writerow(output_row)
                    else:
                        # Submit the API call to the thread pool
                        # executor.submit returns a 'future' object
                        future = executor.submit(get_gemini_response, prompt, api_key)
                        # Store the future and its corresponding row
                        future_to_row[future] = row

                # Now, process the results *as they complete*
                # This is more efficient and saves progress as we go.
                # We wrap as_completed with tqdm to get our progress bar
                
                total_jobs = len(future_to_row)
                
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_row), 
                    total=total_jobs, 
                    desc="Processing prompts", 
                    unit="row"
                ):
                    # Get the original row associated with this completed future
                    original_row = future_to_row[future]
                    
                    try:
                        # Get the result from the completed job
                        response = future.result()
                    except Exception as e:
                        # This catches any unexpected errors from the job itself
                        response = f"ERROR: Job failed with exception: {e}"

                    # Create the new output row and write it
                    output_row = original_row.copy()
                    output_row['response'] = response
                    writer.writerow(output_row)
            
            # --- END PARALLEL BLOCK ---

    except IOError as e:
        print(f"Error writing to output file '{OUTPUT_CSV_FILE}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

    print("\nBatch processing complete.")
    print(f"Results saved to '{OUTPUT_CSV_FILE}'.")


if __name__ == "__main__":
    # To run this script:
    # 1. Install the required libraries:
    #    pip install python-dotenv requests tqdm
    #
    # 2. Save this file as 'batch_processor.py'.
    # 3. Create 'my_labeled_dataset.csv'.
    # 4. Create '.env' file with your GEMINI_API_KEY.
    # 5. Run: python batch_processor.py
    
    process_prompts_from_csv()