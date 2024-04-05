import time
from os import listdir, path

import gradio as gr
import pandas as pd
from openai import OpenAI
from transformers import pipeline

# keeping track of chat histories
chat_history_1 = []
chat_history_2 = []


# counts all numbers of files in a directory
def count_files(directory):
    return len([f for f in listdir(directory) if path.isfile(path.join(directory, f))])


# generates mistral chat response from chat prompts
def mistral_response(chats):

    try:
        # using pipeline
        pipe = pipeline("text-generation", "Open-Orca/Mistral-7B-OpenOrca")

        # getting generated text
        generated_text = pipe(chats, max_new_tokens=1000)[0]['generated_text'][-1]

        # returning generated text
        return generated_text['content']
    except Exception:
        raise gr.Error("Mistral API is running on local machine which encountered an error! Please try again.")


# generates chatgpt api response from chat prompts
def chatgpt_api_response(chats):

    try:

        # need api key from openai
        openai_api_key = "<API KEY>"

        openai_client = OpenAI(api_key=openai_api_key)

        chatgpt_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chats,
            max_tokens=300,  # for response length
            n=1,
            stop=None,
            temperature=0.7  # creativity vs. informativeness
        )

        # return generated text
        return chatgpt_response.choices[0].message.content

    except Exception:
        raise gr.Error("ChatGPT API encountered an error! Please try again.")


# gradio block
with (gr.Blocks(css="style.css", fill_height=True, title="AI Powered Chatbots") as gradio_interface):

    # gradio row
    with gr.Row():

        # gradio markdown component
        gr.Markdown(
            """
            # AI Powered ChatBot
            We are using the Open-Orca/Mistral-7B-OpenOrca fine-tuned model, specifically the Mistral-7B-v0.1 version. We are focusing on using this model to generate SQL queries and later plan to compare the generated responses with those from ChatGPT. For this purpose, please try asking the bot to generate SQL-related responses.
            """)

    # gradio row
    with gr.Row(elem_id="chatbot_container", equal_height=True):

        # gradio column
        with gr.Column(scale=1):
            chatbox1 = gr.Chatbot(
                label="Mistral AI Model Response",
                elem_id="chatbox_1",
                height=540
            )

        # gradio column
        with gr.Column(scale=1):
            chatbox2 = gr.Chatbot(
                label="ChatGPT API Response",
                elem_id="chatbox_2",
                height=540
            )

    # gradio row
    with gr.Row(elem_id="input_container"):

        # gradio group
        with gr.Group():

            # gradio textbox
            input_textbox = gr.Textbox(
                placeholder="Start asking your question...",
                interactive=True,
                show_label=False,
                lines=2,
                autofocus=True,
                elem_id="text_input"
            )

    # gradio row
    with gr.Row(elem_id="generate_btn_container"):

        # gradio column
        with gr.Column(scale=1):
            mistral_btn = gr.Button("Generate Using Mistral AI", elem_id="mistral_btn")

        # gradio column
        with gr.Column(scale=1):
            chatgpt_btn = gr.Button("Generate Using ChatGPT API", elem_id="chatgpt_btn")

    # gradio row
    with gr.Row(elem_id="tool_btn_container"):

        # gradio column
        with gr.Column(scale=1):
            clear_btn = gr.Button("Clear Everything", elem_id="clear_btn")

        # gradio column
        with gr.Column(scale=1):
            save_btn = gr.Button("Save Responses to CSV by Pair", elem_id="save_btn")

    # prepares mistral start ui
    def mistral_start():
        return gr.Button(interactive=False)

    # resets views when mistral response generation is finish
    def mistral_end():
        return gr.Button(interactive=True)

    # prepares chatgpt start ui
    def chatgpt_start():
        return gr.Button(interactive=False)

    # resets views when chatgpt response generation is finish
    def chatgpt_end():
        return gr.Button(interactive=True)

    # handles mistral bot request
    def mistral_bot_handler(message, history):

        # check and clear history if needed
        if len(history) == 0:
            chat_history_1.clear()

        # updating formatted chat history
        chat_history_1.append({"role": "user", "content": message})

        # return message and history
        return "", history + [[message, None]]

    # handles chatgpt bot request
    def chatbot_bot_handler(message, history):

        # check and clear history if needed
        if len(history) == 0:
            chat_history_2.clear()

        # updating formatted chat history
        chat_history_2.append({"role": "user", "content": message})

        # return message and history
        return "", history + [[message, None]]

    # calls mistral generation function and handles final response
    def mistral_response_handler(history):

        # generating text using mistral
        generated_text = mistral_response(chat_history_1)

        # updating formatted chat history
        chat_history_1.append({"role": "assistant", "content": generated_text})

        try:
            # preparing generated text for streaming response
            history[-1][1] = ""
            for character in generated_text:
                history[-1][1] += character
                time.sleep(0.01)
                yield history
        except Exception as e:
            print(e)

    # calls chatgpt generation function and handles final response
    def chatgpt_response_handler(history):

        # getting generated text from chatgpt
        generated_text = chatgpt_api_response(chat_history_2)

        # updating formatted chat history
        chat_history_2.append({"role": "assistant", "content": generated_text})

        # preparing generated text for streaming response
        history[-1][1] = ""
        for character in generated_text:
            history[-1][1] += character
            time.sleep(0.01)
            yield history

    # function to validate text input
    def validate_textbox(message, history):

        # making sure message is not empty
        if len(message) < 1:
            raise gr.Error("Please enter your question in the textbox!")

        # making sure same input is not being entered multiple time
        if len(history) > 0 and history[-1][0] == message:
            raise gr.Error("Please enter a different prompt!")


    # on click mistral generate button call functions sequentially
    mistral_btn.click(

        # validate text input
        validate_textbox, [input_textbox, chatbox1]).success(

        # initiate mistral start ui change
        mistral_start, outputs=mistral_btn
    ).success(

        # initiate bot handler
        mistral_bot_handler, [input_textbox, chatbox1], [input_textbox, chatbox1], queue=False
    ).success(

        # get response and fill chat box
        mistral_response_handler, chatbox1, chatbox1
    ).success(

        # reset ui change
        mistral_end, outputs=mistral_btn
    )

    # on click chatgpt generate button call functions sequentially
    chatgpt_btn.click(

        # validate text input
        validate_textbox, [input_textbox, chatbox2]).success(

        # initiate mistral start ui change
        chatgpt_start, outputs=chatgpt_btn
    ).success(

        # initiate bot handler
        chatbot_bot_handler, [input_textbox, chatbox2], [input_textbox, chatbox2], queue=False
    ).success(

        # get response and fill chat box
        chatgpt_response_handler, chatbox2, chatbox2
    ).success(
        chatgpt_end, outputs=chatgpt_btn
    )

    # clear history
    def clear_history():

        # remove chat histories
        chat_history_1.clear()
        chat_history_2.clear()
        return "", [], []

    # function for formatting purposes to save data into files
    def get_data_by_key(data, key):
        for entry in data:
            if entry[0] == key:
                return entry[1]
            return None

    # function to save data into files
    def save_history(history1, history2):

        # save only if we have history entry
        if len(history1) < 1:
            raise gr.Error(
                "The conversation history is empty! Please make some conversation with both bots and try again!")

        # will store formatted chat history here
        formatted_chat_history = []

        # looping through history
        for item1 in chat_history_1:

            # getting user prompt
            if item1['role'] == 'user':

                # looping through chat history 2
                for item2 in chat_history_2:

                    # making sure to pick content that matches in both list
                    if item2['role'] == 'user' and item1['content'] == item2['content']:
                        # getting the mistral response from history
                        m_response = chat_history_1[chat_history_1.index(item1) + 1]['content'] if len(
                            chat_history_1) > chat_history_1.index(item1) + 1 else ''

                        # getting the chatgpt response from history
                        c_response = chat_history_2[chat_history_2.index(item2) + 1]['content'] if len(
                            chat_history_2) > chat_history_2.index(item2) + 1 else ''

                        # appending into list
                        formatted_chat_history.append({
                            'question': item1['content'],
                            'mistral_response': m_response,
                            'chatgpt_response': c_response
                        })

        # creating a Pandas DataFrame from the list of dictionaries
        df = pd.DataFrame(formatted_chat_history)

        # setting filename to number of files +1
        filename = count_files("GeneratedData/csv") + 1

        # Save the DataFrame to a CSV file with index=False to avoid an extra index column
        df.to_csv(f"GeneratedData/csv/{filename}.csv", index=False)
        df.to_json(f"GeneratedData/json/{filename}.json", index=False)


    # on click event for clear button
    clear_btn.click(clear_history, outputs=[input_textbox, chatbox1, chatbox2], queue=False)

    # on click event for save button
    save_btn.click(save_history, inputs=[chatbox1, chatbox2], queue=False).success(
        lambda text="File saved successfully!": gr.Info(text), inputs=None, outputs=None
    ).success(
        clear_history, outputs=[input_textbox, chatbox1, chatbox2], queue=False
    ).success(
        lambda text="History has been cleared automatically for new prompts!": gr.Info(text), inputs=None, outputs=None
    )

if __name__ == "__main__":

    # using queue for streaming chat
    gradio_interface.queue()

    # launching gradio interface
    gradio_interface.launch()
