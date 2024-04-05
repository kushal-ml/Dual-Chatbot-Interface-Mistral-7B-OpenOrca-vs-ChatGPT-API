## Dual Chatbot Interface

This repository contains a Python script for building a dual chatbot interface using Gradio API. The interface allows users to interact with two different chatbots, Mistral AI 7B and ChatGPT API, simultaneously. Users can input their questions and receive responses from both chatbots side by side. Additionally, there is an option to save the questions along with their corresponding responses for future reference.

### How to Run

+ Clone the repository to your local machine:

  ```bash
    git clone https://github.com/fahomid/Dual-Chatbot-Interface-Mistral-7B-OpenOrca-vs-ChatGPT-API.git
  ```

+ Install the required packages:
  ```bash
    pip install -r requirements.txt
  ```

+ Update the OpenAI API key in the script (openai_api_key variable) with your own API key.
+ Run the script:
  ```bash
    chat.py
  ```
+ Access the interface by opening the provided URL in your web browser.

### Dependencies
+ Gradio: for building the interactive user interface.
+ Pandas: for data manipulation and saving chat history.
+ Transformers: for interacting with the Mistral AI 7B model.
+ OpenAI's ChatGPT API: for interacting with the ChatGPT model.

### Usage
+ Enter your question in the input textbox.
+ Choose whether to generate responses using Mistral AI 7B or ChatGPT API, or both.
+ View the responses from both chatbots in their respective chatboxes.
+ Optionally, click the "Save Responses to CSV by Pair" button to save the questions along with their corresponding responses to a CSV file for future reference.

### Contributions
Contributions to this project are welcome! If you have any suggestions, feature requests, or bug reports, feel free to open an issue or submit a pull request.

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.
