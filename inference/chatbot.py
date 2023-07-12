import argparse
from transformers import set_seed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gradio as gr
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="LZYSaltedFish/chatfish-1b1-sft",
        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum new tokens to generate per response",
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0
    )
    args = parser.parse_args()
    return args


class myPipeline():
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.device = device
        self.eos_token = self.tokenizer.eos_token
    
    def __call__(self,
                 input_text,
                 max_length=1024,
                 do_sample=True,
                 num_beams=1,
                 temperature=10,
                 top_k=50,
                 repetition_penalty=1.0):
        gen_config = GenerationConfig(max_length=max_length,
                                      do_sample=do_sample,
                                      num_beams=num_beams,
                                      temperature=temperature,
                                      top_k=top_k,
                                      repetition_penalty=repetition_penalty)
        query_len = int(max_length * 3 / 4)
        encoded_input = self.tokenizer(input_text,
                                       truncation=True,
                                       max_length=query_len,
                                       return_tensors='pt')

        inputs = encoded_input['input_ids'].to(self.device)
        input_len = inputs.shape[-1]
        output = self.model.generate(inputs,
                                     generation_config=gen_config)
        output = output[0][input_len:]
        response_text = self.tokenizer.decode(output)
        return response_text
        

def get_generator(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config).half()

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    generator = myPipeline(model=model,
                           tokenizer=tokenizer,
                           device='cuda')
    return generator


def get_model_input(message, chat_history):
    def format_input(user_msg, bot_msg):
        # --- CONVERSATION EXAMPLE ---
        # # Human:
        # 西湖有什么景点？

        # # Assistant:
        # 西湖有苏堤、断桥、定海神针、苏堤夜游、月牙泉、人偶综x乐等景点。
        # ...

        # #Human:
        # 哪个景点离余杭区最近？最好旁边有地铁站

        # # Assistant:
        # 西湖周边的景点，离余杭区最近的应该是苏堤、断桥和月牙泉。
        # ...

        return f"# Human:\n{user_msg}\n\n# Assistant:\n{bot_msg}"

    model_input = ''
    for ep in chat_history:
        model_input += format_input(ep[0], ep[1]) + '\n\n'
    model_input += format_input(message, '')
    return model_input


def get_model_response(generator, user_input, args):
    response = generator(user_input,
                         max_length=args.max_length,
                         do_sample=args.do_sample,
                         num_beams=args.num_beams,
                         temperature=args.temperature,
                         top_k=args.top_k,
                         repetition_penalty=args.repetition_penalty)
    return response


def process_response(response, eos_token):
    output = response if eos_token not in response else response[:response.index(eos_token)]
    return output


def main(args):
    generator = get_generator(args.path)
    set_seed(42)
    
    def respond(message, chat_history):
        model_input = get_model_input(message, chat_history)
        output = get_model_response(generator, model_input, args)
        bot_response = process_response(output, generator.eos_token)
        chat_history.append((message, bot_response))
        return "", chat_history
    
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        chatbot.style(height=1080)
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    demo.launch()


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    main(args)