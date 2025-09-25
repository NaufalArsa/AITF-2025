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
    """
    FASE 1: DATA PREPARATION
    ========================
    
    Fungsi ini memproses dataset IMDB menjadi prompts untuk training:
    
    1. Load 25,000 IMDB movie reviews dari Hugging Face datasets
    2. Filter reviews yang memiliki > 200 tokens (hasil: ~24,895 samples)  
    3. Convert setiap review menjadi prompt dengan panjang random 2-8 tokens
    4. Format dataset untuk kompatibilitas dengan PPOTrainer
    
    Input: Config dengan model_name, dataset_name, panjang prompt
    Output: Dataset dengan format torch berisi queries dan input_ids
    """
    # Build a dataset to be used for the training.
    # It is a series of prompts (each with different length chosen randomly)
    # We will use it to generate the responses and compute the rewards.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the IMDB dataset (25,000 movie reviews)
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    
    # Only choose reviews with more than 200 tokens untuk kualitas prompt yang baik
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    # Random prompt length sampler (2-8 tokens)
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        # From each review just keep the first `input_size` tokens, this represents the prompt used to generate the response
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

if __name__ == '__main__':
    """
    MAIN TRAINING PIPELINE
    ======================
    
    Tahapan utama RLHF Training:
    1. Setup dan konfigurasi environment
    2. Load dan setup models (Policy, Reference, Reward)
    3. Training loop dengan 97 iterasi PPO
    4. Save model yang sudah di-fine-tune
    """
    print("Starting RLHF Training for GPT Sentiment Analysis...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # KONFIGURASI PPO
    # ===============
    # learning_rate: Kecepatan belajar model (1.41e-5 cukup konservatif untuk stabilitas)
    # log_with: Logging menggunakan Weights & Biases untuk monitoring
    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=1.41e-5,
        log_with="wandb",
    )

    import wandb
    wandb.init()  # Initialize Weights & Biases logging

    # PERSIAPAN DATA
    # ==============
    # Build dataset dari 24,895 IMDB reviews menjadi prompts
    dataset = build_dataset(config)

    # SETUP MODEL ARCHITECTURE
    # ========================
    # Policy Model: Model yang akan dilatih dan dioptimalkan
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    
    # Reference Model: Model baseline (frozen) untuk mencegah policy collapse
    # Digunakan untuk menghitung KL divergence constraint
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # PPO TRAINER SETUP
    # =================
    # PPOTrainer mengatur semua aspek training termasuk:
    # - Dataloader management (256 samples per batch)
    # - PPO algorithm implementation
    # - Gradient computation dan model updates
    # - KL divergence constraint enforcement
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    # DEVICE CONFIGURATION
    # ====================
    # Pastikan semua komponen menggunakan device yang sama (GPU jika tersedia)
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    # REWARD MODEL SETUP
    # ==================  
    # DistilBERT sentiment classifier sebagai reward model
    # Memberikan reward tinggi untuk sentiment positif, rendah untuk negatif
    # Model ini sudah di-fine-tune khusus untuk IMDB movie reviews
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    # TEST REWARD MODEL
    # =================
    # Demonstrasi cara kerja reward model dengan contoh text
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    
    # Contoh sentiment negatif (akan dapat reward rendah)
    text = "this movie was really bad!!"
    print("Reward Model Test - Negative Text:")
    print(sentiment_pipe(text, **sent_kwargs))

    # Contoh sentiment positif (akan dapat reward tinggi)
    text = "this movie was really good!!"
    print("Reward Model Test - Positive Text:")
    print(sentiment_pipe(text, **sent_kwargs)) # [{'label': 'NEGATIVE', 'score': -2.335047960281372}, {'label': 'POSITIVE', 'score': 2.557039737701416}]

    # GENERATION PARAMETERS
    # =====================
    # Konfigurasi untuk text generation pada setiap iterasi training
    output_min_length = 4   # Minimum panjang response yang di-generate
    output_max_length = 16  # Maximum panjang response yang di-generate
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    # RESPONSE GENERATION CONFIGURATION
    # =================================
    # Parameter untuk mengontrol kualitas dan variasi generated text
    response_generation_kwargs = {
        "min_length": -1,           # Tidak ada minimum length constraint
        "top_k": 0.0,              # Disable top-k sampling (gunakan top-p)
        "top_p": 1.0,              # Top-p sampling dengan p=1.0 (semua token)
        "do_sample": True,         # Enable sampling untuk variasi output
        "pad_token_id": tokenizer.eos_token_id,
    }

    # ================================================================================
    # MAIN TRAINING LOOP - PROXIMAL POLICY OPTIMIZATION (PPO)
    # ================================================================================
    # 
    # Setiap iterasi terdiri dari 3 fase utama:
    # 1. GENERATION: Generate responses dari prompts
    # 2. REWARD COMPUTATION: Hitung reward menggunakan sentiment model  
    # 3. PPO UPDATE: Update model parameters berdasarkan rewards
    #
    # Total: 97 iterasi x 256 samples per batch = 24,895 total training examples
    # ================================================================================
    
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # INPUT: batch berisi 256 prompts dari dataset IMDB
        query_tensors = batch["input_ids"]

        # =================================================================
        # FASE 1: TRAJECTORY GENERATION
        # =================================================================
        # Generate responses untuk setiap query menggunakan current policy
        # Panjang response di-sample random antara 4-16 tokens
        
        response_tensors = []
        for query in query_tensors:
            # Random sampling panjang response untuk variasi
            gen_len = output_length_sampler()
            response_generation_kwargs["max_new_tokens"] = gen_len
            
            # Generate response menggunakan current policy model
            # Returns: concatenated (query + response) tokens
            response = ppo_trainer.generate(query, **response_generation_kwargs)
            
            # Extract hanya bagian response (hapus query dari awal)
            response_tensors.append(response.squeeze()[-gen_len:])
        
        # Convert token responses ke text untuk analisis sentiment
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # =================================================================
        # FASE 2: REWARD COMPUTATION
        # =================================================================
        # Hitung reward menggunakan DistilBERT sentiment classifier
        # Reward tinggi untuk sentiment positif, rendah untuk negatif
        
        # Gabungkan query + response menjadi full text untuk evaluasi
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        
        # Analisis sentiment untuk setiap generated text
        # Output: List of [{'label': 'NEGATIVE', 'score': X}, {'label': 'POSITIVE', 'score': Y}]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

        # Extract reward score (ambil score untuk label 'POSITIVE')
        # Reward positif untuk sentiment positif, negatif untuk sentiment negatif  
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # =================================================================
        # FASE 3: PPO UPDATE STEP - DETAIL IMPLEMENTASI
        # =================================================================
        # 
        # PPO (Proximal Policy Optimization) adalah heart dari RLHF training
        # Berikut detail matematis dan implementasi:
        #
        # 1. POLICY RATIO COMPUTATION:
        #    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        #    - π_θ: Current policy (model yang sedang dilatih)  
        #    - π_θ_old: Reference policy (frozen model)
        #    - Mengukur seberapa berbeda policy baru vs lama
        #
        # 2. ADVANTAGE CALCULATION:
        #    A_t = Q_t - V(s_t)
        #    - Q_t: Action-value dari reward model (sentiment score)
        #    - V(s_t): Predicted state value dari value head model
        #    - A_t: Seberapa baik action dibanding expected value
        #
        # 3. PPO CLIPPED LOSS:
        #    L^CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
        #    - Mencegah policy update yang terlalu agresif
        #    - ε (epsilon) = clipping parameter (default ~0.2)
        #    - Jika r_t > 1+ε atau r_t < 1-ε, maka di-clip
        #
        # 4. VALUE FUNCTION LOSS:
        #    L^VF = (V_θ(s_t) - V_target)²
        #    - Melatih value head untuk predict expected rewards
        #    
        # 5. TOTAL LOSS:
        #    L_total = L^CLIP - c1 * L^VF + c2 * S[π_θ](s_t)
        #    - c1: value function coefficient (~0.5)
        #    - c2: entropy coefficient (untuk exploration)
        #    - S: entropy bonus untuk mencegah policy collapse
        #
        # ================================================================
        
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # INTERNAL PPO COMPUTATIONS (dilakukan di dalam ppo_trainer.step):
        # 1. Compute log probabilities: log π_θ(a_t|s_t)
        # 2. Compute old log probabilities: log π_θ_old(a_t|s_t)  
        # 3. Calculate policy ratio: r_t = exp(log_prob - old_log_prob)
        # 4. Estimate advantages using GAE (Generalized Advantage Estimation)
        # 5. Compute clipped surrogate loss
        # 6. Backpropagation dan parameter update
        # 7. Update value function head
        
        # Log training statistics untuk monitoring
        ppo_trainer.log_stats(stats, batch, rewards)
        
        # SAFETY MECHANISMS:
        # ==================
        # 1. RATIO THRESHOLD: Jika avg(r_t) > 10.0, skip batch
        #    - Warning: "The average ratio of batch (X) exceeds threshold 10.0"
        #    - Mencegah policy collapse dan instability
        #
        # 2. KL DIVERGENCE CONSTRAINT: 
        #    - Batasi KL(π_θ || π_ref) < threshold
        #    - Jika terlalu tinggi, reduce learning rate atau early stop
        #
        # 3. GRADIENT CLIPPING:
        #    - Clip gradients untuk mencegah exploding gradients
        #    - Typical value: max_grad_norm = 1.0

    # ================================================================================
    # MODEL SAVING
    # ================================================================================
    # Setelah 97 iterasi training selesai, save model yang sudah di-fine-tune
    # Model hasil training akan bias terhadap sentiment positif
    
    print("Training completed! Saving fine-tuned model...")
    
    # Save policy model (yang sudah di-update via PPO)
    model.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)
    
    # Save tokenizer untuk consistency
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