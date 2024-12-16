import os
import json
import time
import sys
from threading import Thread, Event
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import ChatCompletionsClient
from openai import OpenAI

client_openai = OpenAI()

# Load environment variables
AZURE_INFERENCE_ENDPOINT = os.environ.get("AZURE_API_ENDPOINT_405B")
AZURE_INFERENCE_CREDENTIAL = os.environ.get("AZURE_API_KEY_405B")

AZURE_INFERENCE_ENDPOINT_70B = os.environ.get("AZURE_API_ENDPOINT_70B")
AZURE_INFERENCE_CREDENTIAL_70B = os.environ.get("AZURE_API_KEY_70B")

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")

if not AZURE_INFERENCE_ENDPOINT or not AZURE_INFERENCE_CREDENTIAL:
    raise EnvironmentError("Please set 'AZURE_API_ENDPOINT_405B' and 'AZURE_API_KEY_405B' environment variables for the 405B model.")

if not AZURE_INFERENCE_ENDPOINT_70B or not AZURE_INFERENCE_CREDENTIAL_70B:
    raise EnvironmentError("Please set 'AZURE_API_ENDPOINT_70B' and 'AZURE_API_KEY_70B' environment variables for the 70B model.")

if not OpenAI.api_key:
    raise EnvironmentError("Please set OPEN AI API Key")

def get_client(model_choice: str) -> ChatCompletionsClient:
    """
    Returns the appropriate Azure Inference client based on the selected model.
    
    Args:
        model_choice (str): Either "70B" or "405B"
    
    Returns:
        ChatCompletionsClient: The configured Azure Inference client
    """
    if model_choice == "70B":
        return ChatCompletionsClient(
            endpoint=AZURE_INFERENCE_ENDPOINT_70B,
            credential=AzureKeyCredential(AZURE_INFERENCE_CREDENTIAL_70B),
        )
    elif model_choice == "405B":
        return ChatCompletionsClient(
            endpoint=AZURE_INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_INFERENCE_CREDENTIAL),
        )
    else:
        raise ValueError("Invalid model choice.")

def display_timer(stop_event):
    """
    Displays a live timer in the format mm:ss.s that updates in place.
    
    Args:
        stop_event (Event): An event to stop the timer.
    """
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        tenths = int((elapsed_time - int(elapsed_time)) * 10)
        timer_display = f"{minutes:02}:{seconds:02}.{tenths}"
        sys.stdout.write(f"\rProcessing Time: {timer_display}")
        sys.stdout.flush()
        time.sleep(0.1)

def query_model(client: ChatCompletionsClient, user_query: str):
    """
    Sends a user-provided query to the LLM model and prints the response.
    
    Args:
        client (ChatCompletionsClient): The initialized model client.
        user_query (str): The prompt to send to the model.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("Query sent to LLM. Waiting for response...")
    
    # Start the timer
    stop_event = Event()
    timer_thread = Thread(target=display_timer, args=(stop_event,))
    timer_thread.start()

    try:
        # Make the API call
        response = client.complete(messages=messages, max_tokens=4000)

        # Stop the timer
        stop_event.set()
        timer_thread.join()
        sys.stdout.write("\n")  # Move to the next line after timer finishes

        # Print response
        response_text = response.choices[0].message.content
        print("Model Response:")
        print(response_text)
    except Exception as e:
        # Stop the timer in case of an error
        stop_event.set()
        timer_thread.join()
        print(f"\nError querying the LLM model: {e}")


def query_openai_model(model_name: str, user_query: str):
 print("Query sent to OpenAI LLM. Waiting for response...")

 stop_event = Event()
 timer_thread = Thread(target=display_timer, args=(stop_event,))
 timer_thread.start()

 try:
    completion = client_openai.chat.completions.create(
    messages=[
        {"role": "user", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_query}
    ],
    model=model_name,
    )

    # Stop the timer
    stop_event.set()
    timer_thread.join()
    sys.stdout.write("\n")  # Move to the next line after timer finishes

    response_text = completion.choices[0].message.content
    print("Model Response:")
    print(response_text)
 except Exception as e:
    stop_event.set()
    timer_thread.join()
    print(f"\nError querying the OpenAI model: {e}")


def main():
    # Display a menu
    print("Please select a model:")
    print("1) 70B Model")
    print("2) 405B Model")
    print("3) o1-preview-2024-09-12")
    print("4) o1-mini-2024-09-12")
    print("5) gpt-4o-2024-08-06  (supports structured response e.g. JSON)")
    print("6) gpt-3.5-turbo-0125")
    
    
    choice = input("Enter your choice (1-5): ").strip()

    if choice == "1":
        model_choice = "70B"
        client = get_client(model_choice)
        is_openai_model = False
    elif choice == "2":
        model_choice = "405B"
        client = get_client(model_choice)
        is_openai_model = False
    if choice == "3":
        model_choice = "o1-preview-2024-09-12"
        is_openai_model = True
        # client_openai = OpenAI()
    elif choice == "4":
        model_choice = "o1-mini-2024-09-12"
        is_openai_model = True
        # client_openai = OpenAI()
    elif choice == "5":
        model_choice = "gpt-4o-2024-08-06"
        is_openai_model = True
        # client_openai = OpenAI()

    elif choice == "6":
        model_choice = "gpt-3.5-turbo-0125"
        is_openai_model = True
        # client_openai = OpenAI()

    else:
        print("Invalid choice. Exiting...")
        return



    # Conditionally initialize the client only for Azure models
    if not is_openai_model:
        client = get_client(model_choice)
        print(f"Using the {model_choice} model (Azure).")
    else:
        print(f"Using the {model_choice} model (OpenAI).")

    # Prompt the user for their query
    print("Enter your prompt/query (Press Ctrl+D or Ctrl+Z followed by Enter to finish):")
    user_query = sys.stdin.read()
    print("This is the user input that has been read in")
    print(user_query)

    # Conditionally query the chosen model
    if not is_openai_model:
        query_model(client, user_query)
    else:
        query_openai_model(model_choice, user_query)


if __name__ == "__main__":
    main()
