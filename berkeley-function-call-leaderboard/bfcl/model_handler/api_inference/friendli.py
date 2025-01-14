import os

from bfcl.model_handler.api_inference.openai import OpenAIHandler
from openai import OpenAI


class FriendliHandler(OpenAIHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.client = OpenAI(
            base_url="https://api.friendli.ai/serverless/v1",
            api_key=os.getenv("FRIENDLI_TOKEN"),
        )
        self.is_fc_model = True
