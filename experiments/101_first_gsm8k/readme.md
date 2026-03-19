Under the project root directory, run the following commands

1. Prepaare Data `python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k`

2. Train the model `bash experiments/101_first_gsm8k/gsm8k_grpo.sh`

3. Merge the FSDP checkpoint `bash experiments/101_first_gsm8k/merge_fsdp_ckpt.sh`

4. Check the quality of the model `python experiments/101_first_gsm8k/quality_check_script.py`

You can use the `gsm8k.ipynb` to inspect the data.