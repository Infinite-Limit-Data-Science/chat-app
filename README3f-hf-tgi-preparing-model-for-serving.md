### Preparing the Model

Text Generation Inference improves the model in several aspects.

1) Quantization

TGI supports bits-and-bytes, GPT-Q, AWQ, Marlin, EETQ, EXL2, and fp8 quantization. To speed up inference with quantization, simply set quantize flag to bitsandbytes, gptq, awq, marlin, exl2, eetq or fp8 depending on the quantization technique you wish to use. When using GPT-Q quantization, you need to point to one of the models [here](https://huggingface.co/models?search=gptq). Similarly, when using AWQ quantization, you need to point to one of [these](https://huggingface.co/models?search=awq) models. 

**To leverage GPTQ, AWQ, Marlin and EXL2 quants, you must provide pre-quantized weights. Whereas for bits-and-bytes, EETQ and fp8, weights are quantized by TGI on the fly**.

https://huggingface.co/docs/text-generation-inference/conceptual/quantization

2) RoPE Scaling

https://huggingface.co/docs/text-generation-inference/basic_tutorials/preparing_model