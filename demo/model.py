from transformers import AutoTokenizer, AutoModelForCausalLM


class GenerationModel:
    def __init__(self, model: str, model_kwargs: dict = None):
        model_kwargs = model_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs, )

    def generate(self, prompt: str, input_dict: dict, num_generation: int = 1, generation_kwargs: dict = None):
        prompt = prompt.format(**input_dict)
        prompt = self.tokenizer.encode(prompt, return_tensors="pt")
        generation_kwargs = generation_kwargs or {}
        generated = self.model.generate(prompt, **generation_kwargs, num_return_sequences=num_generation)
        return self.tokenizer.decode(generated[0])

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
