from transformers import PreTrainedTokenizer, PreTrainedModel, pipeline
from ab.gpt.util.Util import extract_code, extract_hyperparam

extra_instructions = (
    " Use PyTorch for the implementation. Keep the code short. Name the main class of the model \"Net\"."
    " The model code must include default parameters for initialization in the constructor. "
    "Provide only the code. Don't provide any explanation. Remove any text from this reply. "
    "Don't include comments in the code."
)

example_prompt = (
        "Write PyTorch code for an efficient classification model that includes self-attention blocks."
        + extra_instructions
)


class ChatBot:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, keep_memory=False):
        self.show_additional_info = False
        self.model = model
        self.tokenizer = tokenizer
        self.__pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.__keep_memory = keep_memory
        if self.__keep_memory:
            self.__messages = []

    def chat(self, prompt: str, max_len=None, max_words=None, engineer_prompt=True) -> tuple[str, str, str]:
        self.model.eval()
        if engineer_prompt:
            prompt += extra_instructions

        if self.__keep_memory:
            self.__messages.append({"role": "user", "content": prompt})
            in_next = self.__messages
        else:
            in_next = [{"role": "user", "content": prompt}]

        out = self.__pipeline(
            in_next,
            max_new_tokens=max_words,
            do_sample=True,  # Allow Random answer
            max_len=max_len
        )[0]["generated_text"][-1]['content']
        assert isinstance(out, str)
        if self.__keep_memory: self.__messages.append({"role": "assistant", "content": out})
        nn = extract_code(out)
        return nn, extract_hyperparam(out), out
