import torch
import numpy as np
import requests
# from Bio.SeqUtils import CodonUsage # Removed due to import issues
from Bio.SeqUtils import ProtParam
from Bio.Seq import Seq
from Levenshtein import distance as levenshtein_distance # pip install python-Levenshtein

class GENEMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        # Pre-defined codon usage table for Homo sapiens (simplified)
        # In a real scenario, load a full CAI index table.
        self.synonymous_codons = {
            'C': ['TGT', 'TGC'], 'D': ['GAT', 'GAC'], 'S': ['TCT', 'TCG', 'TCA', 'TCC', 'AGC', 'AGT'],
            'Q': ['CAA', 'CAG'], 'M': ['ATG'], 'N': ['AAC', 'AAT'], 'P': ['CCT', 'CCG', 'CCA', 'CCC'],
            'K': ['AAG', 'AAA'], 'STOP': ['TAG', 'TGA', 'TAA'], 'T': ['ACC', 'ACA', 'ACG', 'ACT'],
            'F': ['TTT', 'TTC'], 'A': ['GCA', 'GCC', 'GCG', 'GCT'], 'G': ['GGT', 'GGG', 'GGA', 'GGC'],
            'I': ['ATC', 'ATA', 'ATT'], 'L': ['TTA', 'TTG', 'CTC', 'CTT', 'CTG', 'CTA'],
            'H': ['CAT', 'CAC'], 'R': ['CGA', 'CGC', 'CGG', 'CGT', 'AGG', 'AGA'],
            'W': ['TGG'], 'V': ['GTA', 'GTC', 'GTG', 'GTT'], 'E': ['GAG', 'GAA'], 'Y': ['TAT', 'TAC']
        }

    def calculate_perplexity(self, model, sequence_str):
        """
        Metric: Perplexity
        Question: "Is this sequence statistically likely?"
        """
        # Convert string to tensor indices (assuming simple mapping)
        # You need your specific tokenizer map here. 
        # This is a placeholder map: A:0, C:1, G:2, T:3
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        try:
            indices = [mapping[char] for char in sequence_str]
        except KeyError:
            return float('inf') # Invalid char
            
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        
        model.eval()
        with torch.no_grad():
            # Assuming model takes delta as input, this function is tricky.
            # Usually perplexity is P(Sequence). If your model is conditional (generative),
            # we need the logits it predicts at every step.
            # FOR THIS EXAMPLE: We approximate by checking the mean probability of the chosen tokens
            # if the model allows scoring specific sequences.
            
            # Since your specific DNA_LM takes 'delta' and outputs 'logits', 
            # we can't easily calc perplexity without the specific 'delta' that generated it.
            # We will skip strict perplexity here or return a placeholder if architecture differs.
            return 0.0 # Implementation depends on specific forward pass logic

    def calculate_sequence_recovery(self, generated_seq, reference_seq):
        """
        Metric: Sequence Recovery (Identity)
        Question: "How close is it to the starting point/known gene?"
        """
        if not reference_seq:
            return 0.0
        
        dist = levenshtein_distance(generated_seq, reference_seq)
        max_len = max(len(generated_seq), len(reference_seq))
        identity = (max_len - dist) / max_len
        return identity * 100 # Percentage

    def calculate_codon_usage(self, sequence_str):
        """
        Metric: Codon Usage (GC Content & CAI proxy)
        Question: "Does it use the right biological synonyms?"
        """
        seq_obj = Seq(sequence_str)
        
        # 1. GC Content
        gc_content = (sequence_str.count('G') + sequence_str.count('C')) / len(sequence_str)
        
        # 2. Translate to Protein to check validity
        try:
            protein = seq_obj.translate()
            has_stop_early = '*' in protein[:-1]
            valid_orf = not has_stop_early
        except Exception:
            valid_orf = False
            
        return {
            "GC_Content": round(gc_content, 4),
            "Valid_ORF": valid_orf
        }

    def calculate_foldability_esm(self, sequence_str):
        """
        Metric: TM-Score Proxy (pLDDT)
        Question: "Does the DNA code for a 3D shape?"
        Uses ESMFold API (No local GPU install needed).
        """
        if len(sequence_str) % 3 != 0:
            return {"pLDDT": 0.0, "Note": "Length not divisible by 3"}

        protein_seq = str(Seq(sequence_str).translate()).replace("*", "")
        
        # ESMFold API endpoint
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        
        try:
            response = requests.post(url, data=protein_seq, timeout=10)
            if response.status_code == 200:
                # The API returns PDB format text. 
                # pLDDT scores are in the B-factor column (indices 60-66)
                pdb_text = response.text
                plddts = []
                for line in pdb_text.splitlines():
                    if line.startswith("ATOM"):
                        # Extract B-factor (pLDDT)
                        try:
                            score = float(line[60:66].strip())
                            plddts.append(score)
                        except:
                            pass
                
                avg_plddt = np.mean(plddts) if plddts else 0.0
                return {"pLDDT": round(avg_plddt, 2), "Note": "High confidence > 70"}
            else:
                return {"pLDDT": 0.0, "Note": "API Error"}
        except Exception as e:
            return {"pLDDT": 0.0, "Note": str(e)}

    def calculate_functional_score(self, vc_model, control_state, target_state, sequence_str):
        """
        Metric: Growth Assay Proxy
        Question: "Does the gene actually move the cell state?"
        """
        # Convert string to soft/hard tensor
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        indices = [mapping[s] for s in sequence_str]
        seq_tensor = torch.tensor(indices, dtype=torch.long).to(self.device).unsqueeze(0)
        
        # Create one-hot approximation for the Virtual Cell
        # (Assuming VC takes one-hot or embeddings)
        vocab_size = 4
        one_hot = torch.zeros(1, len(indices), vocab_size).to(self.device)
        for i, idx in enumerate(indices):
            one_hot[0, i, idx] = 1.0
            
        with torch.no_grad():
            # Get predicted delta from Virtual Cell
            pred_state = vc_model(control_state, one_hot)
            
            # Calculate distance (MSE or Cosine)
            mse = torch.nn.functional.mse_loss(pred_state, target_state).item()
            
            # Cosine similarity (1.0 is perfect alignment)
            target_delta = target_state - control_state
            pred_delta = pred_state - control_state
            cosine = torch.nn.functional.cosine_similarity(target_delta, pred_delta).item()
            
        return {"MSE_Loss": mse, "Directionality": cosine}