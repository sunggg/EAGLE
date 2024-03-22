from model.ea_model import EaModel
import torch, numpy as np
import time
from transformers import AutoModel

from transformers import AutoTokenizer

# from fastchat.model import get_conversation_template

base_model_path = "/opt/models/mistral/mixtral-8x7b-instruct-v0.1/"
EAGLE_model_path = "/opt/models/eagle/eagle-mixtral-8x7b-instruct-v0.1/"
# base_model_path = "/opt/models/llama-2/llama-2-13b-chat-hf"
# EAGLE_model_path = "/opt/models/eagle/eagle-llama-2-13b-chat-hf"

model = EaModel.from_pretrained(
    Type="Mixtral",
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.eval()
prompt = "Would you write a song about Colorado for me?"

# use_llama_2_chat = True
# if use_llama_2_chat:
#     conv = get_conversation_template("llama-2-chat")
#     sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
#     conv.system_message = sys_p
#     conv.append_message(conv.roles[0], prompt)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt() + " "

# if use_vicuna:
#     conv = get_conversation_template("vicuna")
#     conv.append_message(conv.roles[0], your_message)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

input_ids = model.tokenizer([prompt]).input_ids

# eagle_model = AutoModel.from_pretrained(EAGLE_model_path)
# eagle_model.eval()
# print(eagle_model(input_ids))

input_ids = torch.as_tensor(input_ids).cuda()
t0 = time.perf_counter()
output_ids, acceptance_lengths = model.eagenerate(
    input_ids, temperature=0.0, max_new_tokens=512
)
output1 = model.tokenizer.decode(output_ids[0])
t1 = time.perf_counter()
elapsed = t1 - t0
num_outputs = len(output_ids[0])
print("Speculative decoding")
print(f"Mean acceptance length: {np.mean(acceptance_lengths)}")
print(f"time: {elapsed:.3f} s ({num_outputs/elapsed:.3f} tok/s)")

t0 = time.perf_counter()
output_ids = model.naive_generate(input_ids, temperature=0.0, max_new_tokens=512)
output2 = model.tokenizer.decode(output_ids[0])
t1 = time.perf_counter()
elapsed = t1 - t0
num_outputs = len(output_ids[0])

print("Vanilla decoding")
print(f"time: {elapsed:.3f} s ({num_outputs/elapsed:.3f} tok/s)")
# print(output1)
# print(output2)
print(len(output1), len(output2))
assert output1 == output2
# Sung: This is for the static batching
# # left padding
# model.eval()
# model.tokenizer.padding_side = "left"
# model.tokenizer.pad_token = model.tokenizer.eos_token
# model.config.pad_token_id = model.config.eos_token_id

# sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

# your_message = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
# conv = get_conversation_template("llama-2-chat")
# conv.system_message = sys_p
# conv.append_message(conv.roles[0], your_message)
# conv.append_message(conv.roles[1], None)
# prompt1 = conv.get_prompt() + " "

# your_message = "Hello"
# conv = get_conversation_template("llama-2-chat")
# conv.system_message = sys_p
# conv.append_message(conv.roles[0], your_message)
# conv.append_message(conv.roles[1], None)
# prompt2 = conv.get_prompt() + " "

# input_s = model.tokenizer([prompt1, prompt2], return_tensors="pt", padding=True).to(
#     "cuda"
# )
# print(input_s.input_ids, input_s.input_ids.shape)
