### 文件解读
**preprocessor_config.json**
``` json
//前处理配置
{
  "chunk_length": 30,
  "feature_extractor_type": "WhisperFeatureExtractor",
  "feature_size": 128,
  "hop_length": 160,
  "n_fft": 400,
  "n_samples": 480000,
  "nb_max_frames": 3000,
  "padding_side": "right",
  "padding_value": 0.0,
  "processor_class": "Qwen2AudioProcessor",
  "return_attention_mask": true,
  "sampling_rate": 16000
}
```