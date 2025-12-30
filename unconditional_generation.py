import esm
import random
import torch
from tqdm import tqdm

# Use a for loop to create strings with the desired number of masked tokens
l = 128
proseq = ""
for p in range(l):
    proseq += "<mask>"

# Loading the model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()               
print(type(model))
model.eval()

# Tokenizing the masked amino acid string
seq = [alphabet.cls_idx] + alphabet.encode(proseq) + [alphabet.eos_idx]
seq_SP = torch.tensor([seq])

# Prints the tokenized string
tokens = ""
for p in list(range(1, len(seq) - 1)):
    tokens += str(p) + ", "
print(f"[" + tokens[0:len(tokens)-2] + "]")

# Generate random positions of choice
positions = list(range(1, len(seq) - 1))
random.shuffle(positions)
print(positions)

print(seq_SP)
for p in tqdm(positions):
    # Get the ESM predictions for position p
    result = model(seq_SP)
    logits_SPA = result["logits"].to(torch.float64)
    # TODO: Change USELESS characterts to zero 
    probs_SPA = logits_SPA.softmax(dim=2)
    if not torch.allclose(probs_SPA.sum(), torch.tensor([l+2], dtype=torch.float64), rtol=0.000000000000001):
        print("\nUncle Lee's Tea thinks you are a FAILURE!")
        print(probs_SPA.sum().item())
        break
    print("\nUncle Lee's Tea thinks you are a SUCCESS!")
    # Sample an amino acid for that position
    prob_A = probs_SPA[0, p, :]
    aa = torch.multinomial(prob_A, 1)
    
    # Change position p to be the sampled amino acid
    seq_SP[0, p] = aa
    print(seq_SP)