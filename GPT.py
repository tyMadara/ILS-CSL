import json
import re

import openai
import requests
from retry import retry
from rich import print as rprint


class GPT:
    def __init__(self, model='gpt-3.5-turbo', temperature=0.7):
        self.model = model
        self.temperature = temperature

    @retry(tries=5, delay=2, backoff=2, jitter=(1, 3), logger=None)
    def chatgpt_QA(self, question, outfile_path=None, quiet=False):
        """
        QA with GPT using the API of steamship
        """
        url = "your url to query LLM"
        config = {
            "model": self.model,
            "max_tokens": 5000,
            "temperature": self.temperature
        }
        response = requests.post(
            url, json={"text": question, "config": config})

        if response.status_code == 200:
            self.answer = response.json()["output"]
            if not quiet:
                rprint(
                    f"✅ [green]Successfully[/green] query the [bold yellow]{self.model}[/bold yellow].")
            if outfile_path:
                with open(outfile_path, "w", encoding="utf-8") as outfile:
                    json.dump(response.json(), outfile,
                              indent=4, ensure_ascii=False)
            result = {"model": self.model,
                      "input": question, "output": self.answer}
            return result
        else:
            rprint(
                f"❌ [red]Failed[/red] to query the [bold yellow]{self.model}[/bold yellow], retrying...")
            raise Exception(f"Failed to query the {self.model} model.")


if __name__ == '__main__':
    prompts = ["You are an expert on the childhood diseases.."]
    chatgpt = GPT("gpt-4", 0.7)
    outputs = chatgpt.chatgpt_QA_multi(prompts)
    output = outputs[0]["output"]
    print(output)
