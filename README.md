# Layer-Shared-Conformer
This is a repository for the layer sharing-based Conformer CTC model, based on NVIDIA Nemo toolkit.
Specifically, this repo includes the 1) Conformer-CTC model (134M) and 2) its variation using Mid-Seq based layer shared Conformer (58M).
You can inference the trained model by your own.
The Conformer-CTC model achieves 2.9% WER on LibriSpeech test-clean dataset and 6.8% WER on test-other dataset, and the layer-shared Conformer achieves 3.0% WER and 7.2% WER on test-clean and test-other datasets, respectively.

The layer-shared Conformer has 15M parameters but has the same number of operations compared to conventional model while inference due to layer reusing.

## Training detail
The models are trained with Sentencepiece tokenizer with 128 tokens using CTC.
The backbone of the Conformer-CTC model has 20 layers with 512 hidden dimension and 2048 ffn dimension as shown in the config files.

## Inference
For inference, this code utilizes inference.py to load the config files, tokenizers, and trained models. You should download the trained model in advance for inference.
