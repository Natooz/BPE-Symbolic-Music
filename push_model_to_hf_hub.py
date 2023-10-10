#!/usr/bin/python3 python

"""
Push model to HF hub
"""

# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md
MODEL_CARD_KWARGS = {
    "license": "apache-2.0",
    "tags": ["music-generation"],
}


if __name__ == "__main__":
    from argparse import ArgumentParser

    from transformers import Seq2SeqTrainer, AutoModelForCausalLM

    from exp_generation import experiments

    parser = ArgumentParser(description="Model training script")
    parser.add_argument("--hub-token", type=str, help="", required=False, default="?")
    args = vars(parser.parse_args())

    for exp_ in experiments:
        for baseline_ in exp_.baselines:
            if baseline_.tokenization_config.bpe_vocab_size != 20000:
                continue
            # Load model
            model_ = AutoModelForCausalLM.from_pretrained(str(baseline_.run_path))

            model_name = f"{exp_.dataset}-{baseline_.tokenization}-bpe{baseline_.tokenization_config.bpe_vocab_size // 1000}k"
            model_card_kwargs = {
                "model_name": model_name,
                "dataset": exp_.dataset,
            }
            model_card_kwargs.update(MODEL_CARD_KWARGS)

            # Push to hub
            trainer = Seq2SeqTrainer(model=model_, args=baseline_.training_config)
            trainer.create_model_card(**model_card_kwargs)
            baseline_.tokenizer.save_params(baseline_.run_path / "tokenizer.conf")
            model_.push_to_hub(
                repo_id=model_name,
                commit_message=f"Uploading {model_name}",
                token=args["hub_token"],
                safe_serialization=True,
            )
            # The trainer does not push the weights as safe tensors
            # Don't forget to upload manually the training results / logs
            # trainer.push_to_hub(f"Uploading {model_name}", **model_card_kwargs)
            # https://github.com/huggingface/transformers/issues/25992
