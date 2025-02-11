# 维持外部接口一致，内部函数封装，逻辑更加简洁

#### 下载模型（手动）
huggingface-cli download --resume-download openai/clip-vit-base-patch32 --local-dir openai/clip-vit-base-patch32 --local-dir-use-symlinks False --resume-download

#### 下载模型（自动）
export HF_ENDPOINT="https://hf-mirror.com"