import torch
import torch.nn as nn
import torchmetrics
from torch.nn.parallel import DistributedDataParallel as DDP
import torchmetrics
import numpy as np
import argparse
import time
import yaml
from box import Box
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from datasets import load_dataset, Audio

from models_ctc import ConformerForCTC


def run(config, name_dataset):
    feature_extractor = AudioToMelSpectrogramPreprocessor(**config.preprocessor)
    tokenizer = Tokenizer.from_file(f"./saved_toks/unigram_{config.tokenizer.vocab_size}_libri.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<pad>"
    )
    asr_model = ConformerForCTC(
        feature_extractor=feature_extractor,
        pad_token_id=tokenizer.pad_token_id,
        config_tokenizer=config.tokenizer,
        config_specaug=config.specaug,
        config_model=config.encoder,
        config_architecture=config.architecture
    )

    asr_model.load_state_dict(torch.load(f"saved_models/{config.name}.pt", map_location="cuda:0"))
    asr_model.eval()

    # load dataset
    asr_dataset_test = load_dataset(
        "csv",
        data_files={"test": f"libri/{name_dataset}.csv"}, 
        column_names=["audio", "transc"]
    )
    asr_dataset_test = asr_dataset_test["test"].cast_column("audio", Audio(sampling_rate=16_000))

    def run_model(batch):
        input_features = [torch.from_numpy(samples["array"]).float() for samples in batch["audio"]]
        length = torch.Tensor([x.size(0) for x in input_features])
        transc = [samples for samples in batch["transc"]]

        start_time = time.time()

        input_features = nn.utils.rnn.pad_sequence(input_features, batch_first=True)
        input_features = input_features.cuda()
        length = length.cuda()

        with torch.no_grad():
            logits = asr_model(audio_signal=input_features, length=length)["logits"]
        
        pred_ids = torch.argmax(logits, dim=-1)
        pred_ids = pred_ids.unique_consecutive(dim=-1)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        process_time = time.time() - start_time
        audio_length = input_features.size(-1)/16000

        batch = {
            "process_time": [process_time],
            "transc": transc,
            "pred_str": pred_str,
            "audio_length": [audio_length],
        }
        return batch
    
    asr_dataset_test = asr_dataset_test.map(run_model, batch_size=1, batched=True, remove_columns=["audio", "sex"])
    process_times = asr_dataset_test["process_time"]
    audio_length = asr_dataset_test["audio_length"]
    pred_str = asr_dataset_test["pred_str"]
    transc = asr_dataset_test["transc"]

    metric_wer = torchmetrics.text.WordErrorRate().cuda()
    wer = metric_wer(pred_str, transc)
    rtf = sum(process_times)/sum(audio_length)
    print("[WER]: ", wer)
    print("[RTF]: ", rtf)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_dataset', type=str, default="test-clean")
    parser.add_argument('--name_config', type=str)
    args = parser.parse_args()

    with open(f"./configs/{args.name_config}.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config)

    run(config, args.name_dataset)
