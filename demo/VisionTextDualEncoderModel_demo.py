
# from transformers import (
#     VisionTextDualEncoderModel, #视觉文本双编码器 模型
#     VisionTextDualEncoderProcessor,#视觉文本双编码器 处理器
#     AutoTokenizer,
#     AutoImageProcessor
# )

# model = VisionTextDualEncoderModel.from_vision_text_pretrained(
#     "openai/clip-vit-base-patch32", "FacebookAI/roberta-base"
# )

# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
# image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
# processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# # save the model and processor
# model.save_pretrained("clip-roberta")
# processor.save_pretrained("clip-roberta")



from transformers import (
    VisionTextDualEncoderModel, #视觉文本双编码器 模型
    VisionTextDualEncoderProcessor,#视觉文本双编码器 处理器
    AutoTokenizer,
    AutoImageProcessor
)

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "facebook/sam-vit-base", "IDEA-Research/grounding-dino-tiny"
)
# processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained("IDEA-Research/grounding-dino-tiny")
image_processor = AutoImageProcessor.from_pretrained("facebook/sam-vit-base")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# save the model and processor
model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")