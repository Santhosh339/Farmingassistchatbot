def generate_response(prompt, model, tokenizer, max_length=100):
    """
    Generates a response to the given prompt using the model and tokenizer.

    Parameters:
    - prompt (str): The user's input.
    - model: The pre-trained language model.
    - tokenizer: The tokenizer associated with the model.
    - max_length (int): The maximum length of the generated response.

    Returns:
    - str: The generated response text.
    """
    # Encode the user's input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate a response from the model
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,  # Use sampling for varied responses
        top_p=0.95,      # Nucleus sampling: keep the top 95% likely words
        top_k=50         # Consider only the top 50 most likely words
    )
    
    # Decode the generated tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process the response to ensure it's clean and relevant
    response = response[len(prompt):].strip()  # Remove the prompt from the start if echoed
    return response
