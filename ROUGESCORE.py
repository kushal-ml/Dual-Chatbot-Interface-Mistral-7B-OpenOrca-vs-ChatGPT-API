import json
import matplotlib.pyplot as plt
import evaluate

# Function to compute ROUGE similarity
def compute_rouge_similarity(predictions, references):
    rouge = evaluate.load('rouge')
    min_length = min(len(predictions), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]

    results = rouge.compute(predictions=predictions, references=references)
    
    rescaled_result = rescale_score(results)

    print(rescaled_result)

def rescale_score(score):
    max_rouge1 = score['rouge1']
    max_rougeL = score['rougeL']
    max_rougeLsum = score['rougeLsum']

    # Calculate the rescaled values
    rescaled_rouge1 = (max_rouge1 * 10) + 0.3
    rescaled_rougeL = (max_rougeL * 10) + 0.3
    rescaled_rougeL = (max_rougeLsum * 10) + 0.3

    # Updating the dictionary with the rescaled values
    score['rouge1'] = rescaled_rouge1
    score['rougeL'] = rescaled_rougeL
    score['rougeLsum'] = rescaled_rougeL

    return score


# Function to generate average response
def generate_average_response(mistral_response, chatgpt_response):
    # Tokenize responses
    mistral_tokens = mistral_response.split()
    chatgpt_tokens = chatgpt_response.split()
    avg_length = (len(mistral_tokens) + len(chatgpt_tokens)) // 2
    # Taking the first half of Mistral response and the second half of ChatGPT response
    avg_response = mistral_tokens[:avg_length] + chatgpt_tokens[avg_length:]
    
    return ' '.join(avg_response)


def clean_response(response):
    response = response.replace('\n', ' ')
    response = response.replace('"', '')
    return response

json_files = ['1.json', '2.json', '3.json', '4.json', '5.json', '6.json', '7.json']

# Lists to store ROUGE similarity scores
rouge_similarity_mistral = []
rouge_similarity_chatgpt = []

counter = 0
for file in json_files:
    counter+=1
    with open('/YourDirectoryToJSONFILE/GeneratedJsonFiles/'+file, 'r') as f:
        json_data = json.load(f)
        for i in range(len(json_data['question'])):
            # Get Mistral and ChatGPT responses
            mistral_response = json_data['mistral_response'][str(i)]
            chatgpt_response = json_data['chatgpt_response'][str(i)]

            mistral_response_cleaned_new = ''
            chatgpt_response_cleaned = []
            average_response_cleaned = []

            
            # Generating average response
            average_response = generate_average_response(mistral_response, chatgpt_response)

            mistral_responses_cleaned = [clean_response(response) for response in mistral_response]
            chatgpt_responses_cleaned = [clean_response(response) for response in chatgpt_response]
            average_responses_cleaned = [clean_response(response) for response in average_response]

            average_response_string = ''.join(average_responses_cleaned)
            mistral_response_string = ''.join(mistral_responses_cleaned)
            chatgpt_response_string = ''.join(chatgpt_responses_cleaned)
            print("Mistral Comparison")
            rouge_mistral = compute_rouge_similarity(mistral_response_string, average_response_string)
            print("Chatgpt comparison")
            rouge_chatgpt = compute_rouge_similarity(chatgpt_response_string, average_response_string)

            # Append ROUGE similarity scores to lists
            rouge_similarity_mistral.append(rouge_mistral)
            rouge_similarity_chatgpt.append(rouge_chatgpt)
