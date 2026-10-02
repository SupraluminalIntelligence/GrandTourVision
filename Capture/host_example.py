"""Examples for already-loaded host model/tokenizer; nothing executes on import.

No model downloads or installs. These examples perform a forward ONLY if the
caller explicitly invokes them with their own existing model and tokenizer.
"""
from capture_activations import ActivationCapture


def make_probe_capture(model, tokenizer, *, prompt, positions, layers,
                       model_name, tokenizer_name, mode, device):
    probe = {key: value.to(device) for key, value in tokenizer(prompt, return_tensors="pt").items()}
    ids = probe["input_ids"][0].detach().cpu().tolist()
    capture = ActivationCapture(model_name=model_name, tokenizer_name=tokenizer_name,
                                prompt=prompt, token_ids=ids,
                                token_texts=[tokenizer.decode([token]) for token in ids],
                                positions=positions, layers=layers, mode=mode)
    return capture, probe


def record_probe(model, capture, probe, *, checkpoint, training_step=None,
                 forward_kwargs=None):
    """Record a fixed eval/no-grad probe; restore all prior module mode flags.

    Do not wrap the caller's gradient-producing training forward in no_grad.
    The capture stores only detached CPU samples; it doesn't change parameters.
    An eval probe reduces dropout noise but is not a deterministic guarantee for
    every model/kernel. Batch order and tokenizer/prompt must be held fixed.
    """
    import torch  # caller's existing environment
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad(), capture.snapshot(checkpoint=checkpoint, training_step=training_step):
            # No return value is retained. Avoid output_hidden_states=True.
            # For compatible decoder models pass {"use_cache": False} explicitly.
            model(**probe, **(forward_kwargs or {}))
    finally:
        for module, training in states:
            module.train(training)


def record_inference(model, tokenizer, *, prompt, positions, layers,
                     model_name, tokenizer_name, device, output_path,
                     checkpoint="inference", forward_kwargs=None):
    capture, probe = make_probe_capture(model, tokenizer, prompt=prompt, positions=positions,
                                      layers=layers, model_name=model_name, tokenizer_name=tokenizer_name,
                                      mode="inference", device=device)
    with capture.installed(model):
        record_probe(model, capture, probe, checkpoint=checkpoint, forward_kwargs=forward_kwargs)
    capture.write(output_path)
