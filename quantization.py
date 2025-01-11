import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.quantization import quantize_dynamic
import time

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, texts):
    model.eval()
    total_loss = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
    perplexity = torch.exp(torch.tensor(total_loss / len(texts)))
    return perplexity.item()

# Function to measure latency
def measure_latency(model, tokenizer, text, n_runs=10):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    model.eval()
    with torch.no_grad():
        # Warm-up runs
        for _ in range(3):
            model(**inputs)

        # Measure latency
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_runs):
            model(**inputs)
        end.record()
        torch.cuda.synchronize()
    latency = start.elapsed_time(end) / n_runs  # in milliseconds
    return latency

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Dataset for evaluation
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial Intelligence is transforming the world.",
    "The future of AI is bright and full of possibilities."
]

# Evaluate baseline (FP32)
print("Evaluating FP32 Model...")
fp32_perplexity = calculate_perplexity(model, tokenizer, texts)
fp32_latency = measure_latency(model, tokenizer, "The future of AI is")
fp32_size = sum(param.nelement() * param.element_size() for param in model.parameters()) / 1e6  # in MB
print(f"FP32 Perplexity: {fp32_perplexity}, Latency: {fp32_latency:.2f} ms, Size: {fp32_size:.2f} MB")

# Apply dynamic quantization
print("\nEvaluating Dynamic Quantized Model...")
dynamic_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
dynamic_perplexity = calculate_perplexity(dynamic_model, tokenizer, texts)
dynamic_latency = measure_latency(dynamic_model, tokenizer, "The future of AI is")
dynamic_size = sum(param.nelement() * param.element_size() for param in dynamic_model.parameters()) / 1e6  # in MB
print(f"Dynamic Quantized Perplexity: {dynamic_perplexity}, Latency: {dynamic_latency:.2f} ms, Size: {dynamic_size:.2f} MB")

# Static Quantization
print("\nEvaluating Static Quantized Model...")
static_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)  # Static quantization setup
static_perplexity = calculate_perplexity(static_model, tokenizer, texts)
static_latency = measure_latency(static_model, tokenizer, "The future of AI is")
static_size = sum(param.nelement() * param.element_size() for param in static_model.parameters()) / 1e6
print(f"Static Quantized Perplexity: {static_perplexity}, Latency: {static_latency:.2f} ms, Size: {static_size:.2f} MB")

# Quantization-Aware Training (QAT) - Replace with your QAT-trained model
# Here, we assume QAT has been set up and trained before
print("\nEvaluating Quantization-Aware Training (QAT) Model...")
# You should replace this with the actual QAT model you train.
qat_model = model.half()  # Simulate QAT with fp16 as an example (you'd use a properly trained QAT model)
qat_perplexity = calculate_perplexity(qat_model, tokenizer, texts)
qat_latency = measure_latency(qat_model, tokenizer, "The future of AI is")
qat_size = sum(param.nelement() * param.element_size() for param in qat_model.parameters()) / 1e6
print(f"QAT Perplexity: {qat_perplexity}, Latency: {qat_latency:.2f} ms, Size: {qat_size:.2f} MB")

# Mixed Precision (fp16) Model
print("\nEvaluating Mixed Precision (FP16) Model...")
fp16_model = model.half()  # Convert to FP16 for mixed precision
fp16_perplexity = calculate_perplexity(fp16_model, tokenizer, texts)
fp16_latency = measure_latency(fp16_model, tokenizer, "The future of AI is")
fp16_size = sum(param.nelement() * param.element_size() for param in fp16_model.parameters()) / 1e6
print(f"FP16 Perplexity: {fp16_perplexity}, Latency: {fp16_latency:.2f} ms, Size: {fp16_size:.2f} MB")

# Summary of Results
print("\nSummary of Results:")
print(f"FP32 - Perplexity: {fp32_perplexity}, Latency: {fp32_latency:.2f} ms, Size: {fp32_size:.2f} MB")
print(f"Dynamic Quantized - Perplexity: {dynamic_perplexity}, Latency: {dynamic_latency:.2f} ms, Size: {dynamic_size:.2f} MB")
print(f"Static Quantized - Perplexity: {static_perplexity}, Latency: {static_latency:.2f} ms, Size: {static_size:.2f} MB")
print(f"QAT - Perplexity: {qat_perplexity}, Latency: {qat_latency:.2f} ms, Size: {qat_size:.2f} MB")
print(f"FP16 - Perplexity: {fp16_perplexity}, Latency: {fp16_latency:.2f} ms, Size: {fp16_size:.2f} MB")
