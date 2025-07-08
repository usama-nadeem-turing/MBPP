import requests
import json

def test_model_connection():
    """
    Test if the model server is running and accessible.
    """
    url = "http://localhost:18000/v1/chat/completions"
    
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": "Give me a one-line Python lambda that reverses a string."}],
        "temperature": 0.2,
        "max_tokens": 64
    }
    
    try:
        print("Testing connection to model server...")
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Model server is running and accessible!")
            print(f"Response: {result}")
            return True
        else:
            print(f"✗ Model server returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to model server. Make sure it's running on localhost:18000")
        return False
    except Exception as e:
        print(f"✗ Error testing model connection: {e}")
        return False

if __name__ == "__main__":
    test_model_connection() 