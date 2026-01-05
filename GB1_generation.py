import esm
import random
import torch
from tqdm import tqdm
from constants import SWISSPROT_FASTA_FILE, DEVICE, GB1_masked
from Bio import SeqIO
import random
from models import SAEGuidedESM
from transformers import AutoTokenizer

# 1. Get a random sequence -> GB1
# 2. Mask the 4 positions on the sequence
# 3. Predict amino acid -> sae guided esm or unguided
# 4. Pick random position to insert it
# 5. Insert amino acid then do 3 until no masks are left
# 6. Return full sequece

def generate(input_seq_SP, model_prob_fn):
    # Generate random positions of choice
    positions = list(range(1, input_seq_SP.size(1) - 1))
    random.shuffle(positions)
    order = []

    # Filter the positions to only include the masked ones
    for p in positions:
        if input_seq_SP[0, p] == 32:
            order.append(p)
    print(order) # should only have four things

    print(input_seq_SP)
    for p in tqdm(order):
        # Get the ESM predictions for position p
        probs_SPA = model_prob_fn(input_seq_SP)
        if not torch.allclose(
            probs_SPA.sum(), 
            torch.tensor([probs_SPA.size(0) * probs_SPA.size(1)], dtype=torch.float64).cuda(),
            rtol=0.000000000000001
        ):
            print("\nUncle Lee's Tea thinks you are a FAILURE!")
            print(probs_SPA.sum().item())
            break
        print("\nUncle Lee's Tea thinks you are a SUCCESS!")
        # Sample an amino acid for that position
        prob_A = probs_SPA[0, p, :]
        aa = torch.multinomial(prob_A, 1)
        # Change position p to be the sampled amino acid
        input_seq_SP[0, p] = aa
        print(input_seq_SP)
    return input_seq_SP


if __name__ == "__main__":

    # Load the unconditional model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    model.to(DEVICE)
    
    def model_prob_fn(input_seq_SP):
        result = model(input_seq_SP)
        logits_SPA = result["logits"].to(torch.float64)
        # TODO: Change USELESS characters to zero
        probs_SPA = logits_SPA.softmax(dim=2)
        return probs_SPA


    # Load the guided model
    model = SAEGuidedESM(layer=33)
    model.to(DEVICE)
    def model_prob_fn(input_seq_SP):
        return model(input_seq_SP)


    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    input_seq_SP = tokenizer([GB1_masked], padding=True, return_tensors="pt")["input_ids"].to(DEVICE)

    output_seq_SP = generate(input_seq_SP, model_prob_fn)
    redesigned_proseq = tokenizer.decode(output_seq_SP[0]).replace("<cls>", "").replace("<eos>", "").replace(" ", "")
    print(redesigned_proseq)