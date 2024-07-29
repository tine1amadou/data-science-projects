from transformers import T5Tokenizer, T5ForConditionalGeneration
# Initialize tokenizer and model from Google's FLAN T5 base variant
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
# List to store the summarization results
output_list = []
for conv in df['Conversation']:
    # Prepare the prompt for summarization
    prompt = f"""
    {conv}
    
    What were the main points in that conversation?
    """
    
    # Tokenize the prompt and generate the summary
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Append the summary to the output list
    output_list.append(summary)
  # Add summaries as a new column in the DataFrame
df['Summary'] = output_list
df.head()
# Print each summary to the console
for summary in df['Summary']:
    print(summary)
  from transformers import pipeline
 
# Initialize the zero-shot classification pipeline with the specified multilingual model
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
# Define labels to classify the conversations into cancellation or other
labels = ['cancellation', 'other']
classification = []
 
# Iterate over each conversation in the DataFrame's 'Conversation' column
for conv in df['Conversation']:
    # Classify the conversation and retrieve the primary label
    result = classifier(conv, labels)
    classification.append(result['labels'][0])
  # Create a new column in the DataFrame to indicate whether the conversation is about cancellation
df["Cancellation"] = [True if cls == 'cancellation' else False for cls in classification]
def cancellation_reasons(df):
    # Check if the conversation is flagged as a cancellation
    if df['Cancellation'] == False:
        return 'None'
    else:
        # Prepare the model prompt with the specific question about cancellation reasons
        prompt = f"""
        {df['Conversation']}
 
        What are the issues that led the client to cancel their subscription?
        """
 
        # Convert the prompt into tokens, feed it into the model, and generate the output
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_new_tokens=50, min_length=20)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
      # Apply the cancellation_reasons function to each row in the DataFrame to extract reasons
df['Cancellation_reasons'] = df.apply(cancellation_reasons, axis=1)
df
# Print non-'None' cancellation reasons to review what issues are leading to cancellations
for reason in df['Cancellation_reasons']:
    if reason != "None":
        print(reason)
