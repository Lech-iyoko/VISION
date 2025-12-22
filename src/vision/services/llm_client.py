# LLM client for language model interactions
print("=== SCRIPT STARTED ===")
import os
from groq import Groq
from dotenv import load_dotenv
print("=== IMPORTS COMPLETE ===")

class GroqClient:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
    
        if api_key:
            print(f"✅ GROQ_API_KEY found (length: {len(api_key)} characters)")
        else:
            print("❌ GROQ_API_KEY not found in .env file.")
        
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not found in .env file.")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        print("✅ Groq Client initialized.")

    def generate_response(self, prompt_text):
        """
        Sends a prompt to Groq and returns the text response.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=1024,
            )
            
            response_text = chat_completion.choices[0].message.content
            return response_text
        
        except Exception as e:
            print(f"❌ Error calling Groq API: {e}")
            return "Sorry, I had trouble thinking of a response."

# --- Test Block ---
if __name__ == "__main__":
    try:
        print("--- STARTING TEST BLOCK ---")
        groq_client = GroqClient()
        test_prompt = "What is the capital of England?"
        print(f"Sending test prompt: '{test_prompt}'")
        
        response = groq_client.generate_response(test_prompt)
        
        print("\n--- Groq's Response ---")
        print(response)
        print("-----------------------")
        
    except Exception as e:
        print(f"❌ An error occurred during the test: {e}")