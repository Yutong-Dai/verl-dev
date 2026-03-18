import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT_PATH = "/fsx/home/yutong/Github/verl/checkpoints/merged_ckpts/verl-tutorial/GSM8K-GRPO-03-16-20-16_global_step_105"
DATA_PATH = "~/data/gsm8k/test.parquet"
NUM_EXAMPLES = 5

tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH)
model = AutoModelForCausalLM.from_pretrained(
    CKPT_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

data = pd.read_parquet(DATA_PATH)
examples = data.head(NUM_EXAMPLES)

for i, row in examples.iterrows():
    prompt_messages = list(row["prompt"])
    question = prompt_messages[0]["content"]
    # ground_truth = row["reward_model"]["ground_truth"]
    answer = row["extra_info"]["answer"]

    text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"{'='*80}")
    print(f"Example {i}")
    print(f"{'='*80}")
    print(f"\nQuestion:\n{question}")
    print('-'*40 + '\n')
    print(f"\nReference Answer:\n{answer}")
    print('-'*40 + '\n')
    print(f"\nModel Response:\n{response}")
    print()

# sample outputs

"""
================================================================================
Example 4
================================================================================

Question:
Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens? Let's think step by step and output the final answer after "####".
----------------------------------------


Reference Answer:
If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.
If she feeds the flock 15 cups of feed in the morning, and 25 cups in the afternoon, then the final meal would require 60-15-25=<<60-15-25=20>>20 cups of chicken feed.
#### 20
----------------------------------------


Model Response:
First, we calculate the total amount of feed needed for all the chickens every day. Wendi has 20 chickens, and she gives each chicken 3 cups of feed per day. So, the total feed for the chickens is:

\[ 20 \text{ chickens} \times 3 \text{ cups/chicken} = 60 \text{ cups} \]

Next, we calculate the amount of feed given in the morning and the afternoon:

- In the morning, Wendi gives 15 cups.
- In the afternoon, Wendi gives 25 cups.

Adding these amounts together gives the total feed given in the day:

\[ 15 \text{ cups} + 25 \text{ cups} = 40 \text{ cups} \]

Therefore, Wendi needs 60 cups of feed in the final meal of the day, but since she only has 40 cups available, she will need an additional:

\[ 60 \text{ cups} - 40 \text{ cups} = 20 \text{ cups} \]

So, Wendi needs to give her chickens 20 more cups of feed in the final meal of the day. #### 20
"""