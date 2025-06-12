import torch
import torch.nn.functional as F
import tiktoken
from ModelGPT2 import GPT

def generate_text(model, prompt, num_return_sequences=4, max_length=32, device='cuda'):
        model.eval()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)  # get probabilities
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # topk sampling for top 50 probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B,1), selecting a token from topk 
                xcol = torch.gather(topk_indices, -1, ix)  # gathering corresponding indices
                xgen = torch.cat((xgen, xcol), dim=1)  # append to sequence
        
        generated_texts = []
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            generated_texts.append(decoded)
            print(f"Sample {i + 1}: {decoded}")

        
        return generated_texts
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"running with {device}")

# Extract config and create model
checkpoint_path = 'log/model_final.pt'

print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path,map_location=device)
model_config = checkpoint['config']
model_config.vocab_size = 50304 #for computational effciency(power of 2)
model = GPT(model_config)

#Load model state dict
model.load_state_dict(checkpoint['model'])
model.to(device)



prompt = "Hello, I'm a language model,"

generated_texts = generate_text(
        model=model,
        prompt=prompt,
        num_return_sequences=4,
        max_length=32,
        device=device
    )
