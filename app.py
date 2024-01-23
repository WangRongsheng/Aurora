import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from peft import PeftModel
import time

model_name_or_path = "./models/Mixtral-8x7B-Instruct-v0.1"
lora_weights = "./models/Aurora-Plus/final-checkpoint"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model0 = AutoModelForCausalLM.from_pretrained(model_name_or_path, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(
    model0,
    lora_weights,
)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [0,]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def convert_history_to_text(history):
    text = ""
    if len(history) > 1:
        text = "<s> " + "".join(
                [
                    "".join(
                        [
                            f"[INST]{item[0]}[/INST] {item[1]} ",
                        ]
                    )
                    for item in history[:-1]
                ]
            ) + "</s> "
    text += "".join(
        [
            "".join(
                [
                    f"[INST]{history[-1][0]}[/INST]",
                ]
            )
        ]
    )
    return text

def predict(message, history, max_new_tokens, top_p, top_k, temperature):

    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = convert_history_to_text(history_transformer_format)

    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    t1 = time.time()
    count = 0
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            count += 1
            yield partial_message
    t2 = time.time()
    speed = count/(t2-t1)
    print("inference speed: %f tok/s" % speed)

max_new_tokens = gr.Slider(0, 32768, value=4096, step=1.0, label="Max new tokens", interactive=True)
top_p = gr.Slider(0, 1, value=0.95, step=0.01, label="Top P", interactive=True)
top_k = gr.Slider(0, 2000, value=1000, step=1, label="Top K", interactive=True)
temperature = gr.Slider(0, 1, value=1.0, step=0.01, label="Temperature", interactive=True)
    
gr.ChatInterface(predict, additional_inputs=[max_new_tokens, top_p, top_k, temperature], chatbot=gr.Chatbot(height=600), title="Aurora-Mixtral-8x7B").queue().launch(share=False, server_name="0.0.0.0", server_port=80)
