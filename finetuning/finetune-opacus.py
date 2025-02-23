#change log: 


#### To change -> change to 8B model
# Already changed


# Basics
## Bumped window size to 512 (328 is avg size)
## tried r=8 and lora_alpha = 16. 
## lora_dropout 0.05
## epochs = 4
## learning rate -> 2e-4
## increased weights to be finetuned -> target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]

# DP -> Currently Off
# To activate uncomment lines: 176-185 | 233-234 | 226-227


# -------------------------------
# 0. Check NumPy Version
# -------------------------------
from random import sample
import numpy as np
print("NumPy version:", np.__version__)  # Should print "1.23.5"

# -------------------------------
# 1. Imports and Device Setup
# -------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from opacus import PrivacyEngine
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType

# Set up device: use GPU if available (or MPS if on Apple Silicon) otherwise CPU.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 2. Load Pretrained Model and Tokeni`zer
# -------------------------------
# Here we use EleutherAI/gpt-neo-2.7B, which is a 2.7B model.

#change to 8b where required
model_name = "meta-llama/Llama-3.2-1B"

#getting the tokenizer and and the model for the specific model we care about

# This line downloads (if needed) and initializes a tokenizer using the identifier stored in model_name. 
# The tokenizer converts text into a numerical format (tokens) that the model can process, 
# and it also handles the reverse process (converting tokens back to human-readable text).
tokenizer = AutoTokenizer.from_pretrained(model_name)

# This line loads a pre-trained causal language model (such as GPT-style models) using the same model identifier. 
# It retrieves the model architecture and its pre-trained weights so you can use it for tasks like text generation.
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure a pad token exists (set to eos token if not present).
# 1. Check for the padding token id. If none, use the eos_token as the padding token
if tokenizer.pad_token_id is None:
    print("pad, token doesnt exists, using EOS token")
    tokenizer.pad_token = tokenizer.eos_token

# Adjusts the model's token embedding matrix to match the size of the tokenizer's vocabulary. 
# This is important because adding or changing tokens (like defining a pad token)
# may change the size of the vocabulary, and the model's embedding layer needs to reflect that change.
model.resize_token_embeddings(len(tokenizer))

# -------------------------------
# 3. Apply LoRA for Parameter-Efficient Fine-Tuning
# -------------------------------
# Enable gradient checkpointing to save memory.

# This technique reduces memory usage during training by not storing all intermediate activations 
# during the forward pass. Instead, it saves only a subset of them and recomputes the missing ones 
# during the backward pass.
model.config.gradient_checkpointing = True

# Configure LoRA: update only a small set of additional parameters.
# tried r=4 and lora+alpha = 32. Maybe that destabilized training so modifying to 8 and 16 respectively
#initally was 0.1, changing to 0.05

# studies say best to apply Lora to all layers
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Fine-tuning for causal language modeling.
    inference_mode=False,          # Training mode.
    r=8,                           # Rank of low-rank decomposition.
    lora_alpha=16,                 # Scaling factor.
    lora_dropout=0.05,               # Dropout rate for LoRA layers.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
)

# To get all the intermediate layer config of the model
# for name, module in model.named_modules():
#     print(name, ":", module)

# This function call takes the pre-trained model and applies the LoRA configuration you defined. 
# It modifies the model so that, instead of updating all parameters during fine-tuning, 
# only a small subset (the LoRA adapters) is trained.
model = get_peft_model(model, lora_config)
print("LoRA applied. Trainable parameters:")
model.print_trainable_parameters()

# Move the model to the chosen device and set to training mode.
model.to(device)
model.train()

# -------------------------------
# 4. Prepare Training Data
# -------------------------------
# For demonstration, we use a small synthetic training set.
# Replace train_texts with your actual dataset.

import json

# Initialize an empty list to store the formatted strings
formatted_strings = []

# Open the JSONL file and read each line
with open("finetuning/train.jsonl", "r") as f:
    j = 0
    for line in f:
        # Parse the JSON data from the line
        data = json.loads(line.strip())
        # Extract values
        rating = data['Rating']
        title = data['Title']
        review = data['Review']
        
        # Format the string as per the required format
        formatted_string = f'"System prompt : Given the Rating and Title, you are required to generate the review" | "Rating": {rating} | "Title": {title} | "Review": {review}'
        
        # Add the formatted string to the list
        formatted_strings.append(formatted_string)

# Now `formatted_strings` contains the list of strings in the desired format
print("Size: ",len(formatted_strings))
print(formatted_strings[0])
train_texts = formatted_strings
strs = [len(formatted_str) for formatted_str in formatted_strings]
print("length of largets string is: ",sum(strs) / len(strs))
#avg around 328

# Tokenize training texts with padding and truncation.
encodings = tokenizer(train_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# For causal language modeling, use input_ids as labels.
# Replace pad token positions with -100 so that they are ignored by the loss.

#creates a copy of your input IDs, so you can modify them without affecting the original tensor.
labels = input_ids.clone()

#replaces all padding token positions with -100. This is a common convention (especially with PyTorch’s CrossEntropyLoss) 
# to indicate that these positions should be ignored during loss computatio
labels[input_ids == tokenizer.pad_token_id] = -100

print("Training data shape:", input_ids.shape)

# Create a TensorDataset and DataLoader with a small batch size.
train_dataset = TensorDataset(input_ids, attention_mask, labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

# -------------------------------
# 5. Set Up Optimizer and PrivacyEngine (DP-SGD)
# -------------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)


# DP-SGD commented out for now

# privacy_engine = PrivacyEngine()
# model, optimizer, train_loader = privacy_engine.make_private(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_loader,
#     noise_multiplier=1.0,      # Adjust for your desired privacy guarantee.
#     max_grad_norm=1.5,         # Gradient clipping norm.
#     batch_first=True,
#     loss_reduction="mean",
#     poisson_sampling=False     # Use standard sampling.
# )

# -------------------------------
# 6. Fine-Tune the Model with DP-SGD
# -------------------------------
model.train()
epochs = 4  # Use more epochs for a real task.
for epoch in range(epochs):
    total_loss = 0.0
    i = 0
    for batch in train_loader:
        if i%50 == 0:
            print(i)
        i+=1
        # Move each element of the batch to the device.
        input_ids_batch, attention_mask_batch, labels_batch = [x.to(device) for x in batch]
        
        # Create a position_ids tensor: shape [batch_size, seq_len]
        seq_len = input_ids_batch.size(1)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(input_ids_batch.size(0), 1)
        
        # Forward pass: compute the loss.
        outputs = model(
            input_ids=input_ids_batch,
            attention_mask=attention_mask_batch, 
            position_ids=position_ids,
            labels=labels_batch
        )
        loss = outputs.loss
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Average loss: {avg_loss:.4f}")



#get the privacy budget
# epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
# print(f"Achieved privacy budget: ε = {epsilon:.2f}")

# -------------------------------
# 7. Evaluation
# -------------------------------
# Remove DP hooks to restore the underlying model.
# model.remove_hooks()
# model = model._module  # Unwrap the model.

# Specify the directory where you want to save your fine-tuned model
save_directory = "./finetuned_model_dp"

# Save the model weights and configuration
model.save_pretrained(save_directory)

# Save the tokenizer (this ensures that any custom tokens are preserved)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")

model.eval()

# Evaluate using a sample prompt.
while True:
    sample_prompt = input("Input: ")
    if sample_prompt.lower() == "bye":
        break
    enc = tokenizer(sample_prompt, return_tensors='pt', padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    generated_ids = model.generate(**enc, max_length=512, do_sample=True, top_k=50)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Generated text:", generated_text)

    #lora_alpha = 4
    #reduce loss to 2e-5

