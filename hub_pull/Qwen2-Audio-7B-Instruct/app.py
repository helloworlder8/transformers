import gradio as gr
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from argparse import ArgumentParser
import os

DEFAULT_CKPT_PATH = 'Qwen/Qwen2-Audio-7B-Instruct'


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8520,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def add_text(chatbot, task_history, text, audio):
    content = []
    if text:
        content.append({'type': 'text', 'text': text})
    if audio and os.path.exists(audio):
        content.append({'type': 'audio', 'audio_url': audio})

    if not content:
        # 如果没有输入，不更新聊天记录
        return chatbot, task_history

    task_history.append({"role": "user", "content": content})

    # 显示输入的文本和音频（如果有）
    display_text = text if text else "(No text input)"
    display_audio = f"(Audio uploaded: {os.path.basename(audio)})" if audio else ""

    chatbot.append([f"{display_text} {display_audio}", None])  # 用户输入的内容在左边
    return chatbot, task_history


def reset_user_input():
    """重置用户输入字段。"""
    return gr.update(value="")


def reset_state():
    """重置聊天记录。"""
    return [], []


def regenerate(chatbot, task_history):
    """重新生成上一个机器人回复。"""
    if task_history and task_history[-1]['role'] == 'assistant':
        task_history.pop()
        chatbot.pop()
    if task_history:
        chatbot, task_history = predict(chatbot, task_history)
    return chatbot, task_history


def predict(chatbot, task_history):
    """从模型生成回复。"""
    # print(f"{task_history=}")
    # print(f"{chatbot=}")
    text = processor.apply_chat_template(task_history, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in task_history:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audio_path = ele.get('audio_url')
                    if audio_path and os.path.exists(audio_path):
                        audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
                        audios.append(audio)

    if len(audios) == 0:
        audios = None
    # print(f"{text=}")
    # print(f"{audios=}")
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    if not args.cpu_only:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"{response=}")
    task_history.append({'role': 'assistant',
                         'content': response})
    chatbot.append((None, response))  # 将回复添加到聊天记录中
    return chatbot, task_history


def _launch_demo(args):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label='Audio-Chat', elem_classes="control-height", height=350)  # 聊天机器人控制组件
        task_history = gr.State([])

        # 用户输入组件
        with gr.Row():
            with gr.Column(scale=4):
                user_text = gr.Textbox(
                    label="Enter Text",
                    placeholder="Type your message here...",
                    lines=8
                )
            with gr.Column(scale=1):
                user_audio = gr.Audio(
                    sources=["upload", "microphone"],  # 支持上传和麦克风输入
                    type="filepath",
                    label="Upload Audio or Record via Microphone",
                )
        submit_button = gr.Button("🚀 Submit (发送)")

        # 控制按钮
        with gr.Row():
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        # 提交逻辑
        submit_button.click(
            fn=add_text,
            inputs=[chatbot, task_history, user_text, user_audio],
            outputs=[chatbot, task_history]
        ).then(
            predict,
            inputs=[chatbot, task_history],
            outputs=[chatbot, task_history],
            show_progress=True
        )
        submit_button.click(reset_user_input, [], [user_text])

        # 清除历史
        empty_bin.click(
            reset_state,
            inputs=[],
            outputs=[chatbot, task_history],
            show_progress=True
        )

        # 重新生成回复
        regen_btn.click(
            regenerate,
            inputs=[chatbot, task_history],
            outputs=[chatbot, task_history],
            show_progress=True
        )

    demo.queue().launch(
        share=False,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


if __name__ == "__main__":
    args = _get_args()
    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    model.generation_config.max_new_tokens = 2048  # For chat.
    print("generation_config", model.generation_config)
    processor = AutoProcessor.from_pretrained(args.checkpoint_path, resume_download=True)
    _launch_demo(args)
