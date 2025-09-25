"""
Reinforcement Learning from Human Feedback (RLHF) Training for GPT Sentiment Analysis
===================================================================================

Mekanisme Training Komprehensif:

1. ARSITEKTUR SISTEM:
   - Policy Model: Model GPT-2 yang akan dilatih dan dioptimalkan
   - Reference Model: Model baseline (frozen) untuk mencegah policy collapse
   - Reward Model: DistilBERT sentiment classifier untuk memberikan feedback
   - PPO Trainer: Mengatur proses optimization menggunakan algoritma PPO

2. ALUR TRAINING (97 Iterasi):
   Phase 1: Generation - Generate responses dari prompts (256 samples per batch)
   Phase 2: Reward Computation - Analisis sentiment menggunakan reward model
   Phase 3: PPO Update - Optimisasi model berdasarkan rewards
   
3. MATEMATIS PPO - DETAIL IMPLEMENTASI:
   a) Policy Ratio Computation:
      r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
      = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))
      
   b) Advantage Estimation (GAE):
      A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
      dimana δ_t = r_t + γV(s_{t+1}) - V(s_t)
      
   c) PPO Clipped Objective:
      L^CLIP = E_t[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
      
   d) Value Function Loss:
      L^VF = E_t[(V_θ(s_t) - V_target)²]
      
   e) Total Loss:
      L = L^CLIP - c1*L^VF + c2*S[π_θ](s_t)

4. PPO IMPLEMENTATION LOCATION:
   - External: TRL library (`trl.trainer.PPOTrainer`)
   - Internal: `ppo_trainer.step()` method
   - Core algorithm: Handled by Accelerate + PyTorch

5. SAFETY MECHANISMS - MENGAPA DIPERLUKAN:
   - Policy Collapse: Tanpa clipping, policy bisa berubah drastis
   - Instability: Large policy updates → oscillating rewards  
   - Sample Efficiency: PPO lebih sample efficient daripada REINFORCE
   - Robustness: Lebih robust terhadap hyperparameter choices
"""

import torch
from tqdm import tqdm

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

if __name__ == '__main__':
    print("Starting RLHF Training for GPT Sentiment Analysis...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=1.41e-5,
        log_with="wandb",
    )

    import wandb
    wandb.init()

    dataset = build_dataset(config)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=0 if torch.cuda.is_available() else -1)

    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 32}

    output_min_length, output_max_length = 4, 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    response_generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        gen_lens = [output_length_sampler() for _ in query_tensors]
        response_tensors = [
            ppo_trainer.generate(query, max_new_tokens=gen_len, **response_generation_kwargs).squeeze()[-gen_len:]
            for query, gen_len in zip(query_tensors, gen_lens)
        ]
        batch["response"] = [tokenizer.decode(r) for r in response_tensors]
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    print("Training completed! Saving fine-tuned model...")
    model.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)
    tokenizer.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)
    print("Model saved to: ./gpt2-imdb-pos-v2/")
    print("RLHF Training completed successfully!")  
    
    # ================================================================================
    # HASIL TRAINING:
    # ================================================================================
    # Model yang dihasilkan akan memiliki karakteristik:
    # ✅ Bias positif - cenderung generate text dengan sentiment positif
    # ✅ Context aware - memahami konteks movie reviews  
    # ✅ Fluency preserved - tetap generate text yang natural dan coherent
    # ✅ Reward optimized - dioptimalkan untuk memaksimalkan positive sentiment score
    # 
    # Penggunaan: Load model dengan transformers dan gunakan untuk text generation
    # yang akan menghasilkan output dengan bias sentiment positif
    # ================================================================================