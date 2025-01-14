import os

from bfcl.model_handler.api_inference.openai import OpenAIHandler
from openai import OpenAI


class FriendliHandler(OpenAIHandler):
    FSE_MODELS = [
        "meta-llama-3.1-8b-instruct",
        "meta-llama-3.1-70b-instruct",
        # Add additional model names for fse here.
    ]

    def __init__(self, model_name, temperature) -> None:
        """
        e.g. model_name = 
            - "meta-llama-3.1-8b-instruct" (fse) -> "https://api.friendli.ai/serverless/v1"
            - "450abXXVVXX" (fde) -> "https://api.friendli.ai/dedicated/v1"
            - "450abXXVVXX:tdlr" (fde lora adapter) -> "https://api.friendli.ai/dedicated/v1"
            - "6000" (localhost fc) -> "http://localhost:6000/v1"
            - "http://1.1.1.1:8000" (remote fc) -> "http://1.1.1.1:8000/v1"
        """

        # Call the routing function
        model_id, base_url = self.route_model(model_name)

        super().__init__(model_id, temperature)
        
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("FRIENDLI_TOKEN"),
        )
        self.is_fc_model = "localhost" in base_url or "http" in base_url

    @staticmethod
    # TODO: Friendli Container Lora Adapter case is not handled
    def route_model(model_name: str):
        """
        Determine the base_url based on the model name.
        
        Args:
            model_name (str): The model identifier.
        
        Returns:
            tuple: (model_name, base_url)
        """
        if model_name in FriendliHandler.FSE_MODELS:  # Friendli Serverless Endpoints
            base_url = "https://api.friendli.ai/serverless/v1"
        elif model_name.isdigit():  # Friendli Container - model name is a numeric string
            base_url = f"http://localhost:{model_name}/v1"
        elif model_name.startswith("http://"):  # Friendli Container - full URL
            base_url = f"{model_name}/v1"
        else:  # Dedicated Endpoints or Dedicated Endpoints lora adapter
            base_url = "https://api.friendli.ai/dedicated/v1"

        return model_name, base_url
