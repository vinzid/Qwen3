import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional


app = FastAPI()


class MessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None


class MessageOutput(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class Choice(BaseModel):
    message: MessageOutput 


class Request(BaseModel):
    messages: List[MessageInput]


class Response(BaseModel):
    model: str
    choices: List[Choice]


@app.post("/v1/chat/completions", response_model=Response)
async def create_chat_completion(request: Request):
    global model, tokenizer

    print(datetime.now())
    print("\033[91m--received_request\033[0m", request)
    text = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128000
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    end_of_think = content.find("</think>")
    if end_of_think != -1:
        content = content[end_of_think + 8:]
    print(datetime.now())
    print("\033[91m--generated_text\033[0m", content)

    message = MessageOutput(
        role="assistant",
        content=content,
    )
    choice = Choice(
        message=message,
    )
    response = Response(model=sys.argv[1].split('/')[-1].lower(), choices=[choice])
    return response


torch.cuda.empty_cache()

if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
