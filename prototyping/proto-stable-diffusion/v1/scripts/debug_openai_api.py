import requests
import ipdb
import openai

# Don't leave tokens in commit-able code. See gitignored tokens.py instead.
OPENAI_TOKEN = None
client = openai.OpenAI(api_key=OPENAI_TOKEN)

def generate_image_openai(prompt, output_path="image.png"):

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url"
        )
        image_url = response.data[0].url
        img_data = requests.get(image_url).content
        with open(output_path, "wb") as f:
            f.write(img_data)
        print(f"Image saved to {output_path}")
        #return output_path
    except openai.OpenAIError as e:
        print("OpenAI error:", e)
        #return None

prompt = "A serene mountain landscape with a lake, in the style of Studio Ghibli"

print("Please be careful. It costs .08$ to make one call.")
ipdb.set_trace()

generate_image_openai(prompt)