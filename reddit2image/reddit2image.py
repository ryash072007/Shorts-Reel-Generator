from groq import Groq
import json, os

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

history = []


def get_text_reply(prompt, add_to_memory=False):
    global messages

    chat_completion = client.chat.completions.create(
        messages=history
        + [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gemma2-9b-it",
    )

    if add_to_memory:
        history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        history.append(
            {"role": "system", "content": chat_completion.choices[0].message.content}
        )

    return chat_completion.choices[0].message.content


def init_text_model():
    global messages
    prompt = """Prompt:

Break down the following text into smaller and smaller sections while preserving meaning. Each section should be paired with an AI-generated image prompt that visually represents its content. Ensure that:

Each section retains logical coherence and contributes to the full understanding of the original passage.
The sum of all excerpts should reconstruct the original text when combined.
Each excerpt is paired with a corresponding image prompt that describes what an AI should generate to visualize the excerpt’s meaning.
Output the result in JSON format with the following structure:

json
{
  "breakdown": [
    {
      "excerpt": "[Excerpt from the passage]",
      "image_prompt": "[Detailed AI image prompt describing a visual representation of the excerpt]"
    },
    {
      "excerpt": "[Next excerpt]",
      "image_prompt": "[AI image description]"
    }
    // Continue this format until the entire passage is broken down
  ]
}
Ensure that the AI image prompts are descriptive and suitable for text-to-image models."""
    reply = get_text_reply(prompt, True)


def convert_output_to_dict(output):
    curly_1 = output.find("{")
    curly_2 = output.rfind("}")
    output = output[curly_1 + 1 : curly_2]
    output = "{" + output + "}"

    output_dict = json.loads(output)
    return output_dict


def convert_output_to_image(output_dict):
    prompts = []
    for item in output_dict["breakdown"]:
        prompts.append(item["image_prompt"])
    return prompts


if __name__ == "__main__":

    init_text_model()

    reply = get_text_reply(
        """Vancouver, Colony of the British Empire

June 17, 1859

Rear Admiral Robert Baines was drowning.

His body—battle-hardened, scarred, yet still strong—was sinking deeper and deeper into the abyss of depression. His wife had long left him for a nineteen-year-old crypto entrepreneur, and his son had become a YouTube prankster. What a disgrace…

Only the service remained, but even here, in the seemingly familiar embrace of the Royal Army, he suffocated. Endless drills, reports, formations—it all felt like a slow death. His soul craved fierce battles and glorious victories, the enemy’s blood on his bayonet, the cold wind on his face, and the exhilarating roar of cannon fire."""
    )

    data = convert_output_to_dict(reply)
    prompts = convert_output_to_image(data)
