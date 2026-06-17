from .qwen3 import KVCache
import warnings
import torch


def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")

    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("Using Intel GPU")

    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

@torch.inference_mode()
def generate_text_basic(model, token_ids, max_new_tokens, eos_token_id=None):
    input_length = token_ids.shape[1]
    model.eval()

    logits, _ = model(token_ids)
    out = logits[:, -1]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        token_ids = torch.cat([token_ids, next_token], dim=1)
        logits, _ = model(token_ids)
        out = logits[:, -1]

    return token_ids[:, input_length:]

# Previously, chapter 2 used a non-streaming function, and the
# *_stream functions were introduced in the exercises
# While the simpler generate_text_basic_cache is not used anymore
# it is kept here for backwards compatibility reasons
@torch.inference_mode()
def generate_text_basic_cache(model, token_ids, max_new_tokens, eos_token_id=None):
    input_length = token_ids.shape[1]
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    logits, cache = model(token_ids, cache=cache)
    out = logits[:, -1]
    generated_tokens = []

    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        generated_tokens.append(next_token)
        logits, cache = model(next_token, cache=cache)
        out = logits[:, -1]

    if generated_tokens:
        return torch.cat(generated_tokens, dim=1)
    return token_ids[:, input_length:]

@torch.inference_mode()
def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()

    logits, _ = model(token_ids)
    out = logits[:, -1]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        yield next_token
        token_ids = torch.cat([token_ids, next_token], dim=1)

        logits, _ = model(token_ids)
        out = logits[:, -1]

@torch.inference_mode()
def generate_text_basic_stream_cache(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    kv_cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    logits, new_caches = model(token_ids, cache=kv_cache)
    kv_cache.cache = new_caches   # update KVCache with list
    out = logits[:, -1]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

        yield next_token
        logits, new_caches = model(next_token, cache=kv_cache)
        kv_cache.cache = new_caches   # update again
        out = logits[:, -1]

    # return token_ids[:, input_length:]

def _print_peak_memory_stats(device):
    device = torch.device(device)
    backend = getattr(torch, device.type, None)
    if backend is None or not hasattr(backend, "is_available"):
        return
    if not backend.is_available():
        return

    sync_fn = getattr(backend, "synchronize", None)
    if callable(sync_fn):
        try:
            sync_fn(device=device)
        except TypeError:
            sync_fn()

    try:
        max_mem_bytes = backend.max_memory_allocated(device=device)
    except TypeError:
        max_mem_bytes = backend.max_memory_allocated()

    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max {device.type.upper()} memory allocated: {max_mem_gb:.2f} GB")

    try:
        backend.reset_peak_memory_stats(device=device)
    except TypeError:
        backend.reset_peak_memory_stats()


def generate_stats(output_token_ids, tokenizer, start_time,
                   end_time):
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")

    for name, backend in (("CUDA", getattr(torch, "cuda", None)),
                          ("XPU", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():

            # Check whether we are actually using this backend
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )

            # Synchronize if supported (important for async backends)
            if hasattr(backend, "synchronize"):
                backend.synchronize()

            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()
