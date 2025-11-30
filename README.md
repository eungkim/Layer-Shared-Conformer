# Layer-Shared-Conformer
This is a repository for the layer sharing-based Conformer CTC model, based on NVIDIA Nemo toolkit.
Specifically, this repo includes the 1) Conformer-CTC model (34M) and 2) its variation using Mid-Seq based layer shared Conformer (15M).
You can inference the trained model by your own.
The Conformer-CTC model achieves 3.5% WER on LibriSpeech test-clean dataset and 8.5% WER on test-other dataset, and the layer-shared Conformer achieves 3.9% WER and 9.7% WER on test-clean and test-other datasets, respectively.

The layer-shared Conformer has 15M parameters but has the same number of operations compared to conventional model while inference due to layer reusing.

## Training detail
The models are trained with Sentencepiece tokenizer with 128 tokens using CTC.
The backbone of the Conformer-CTC model has 20 layers with 256 hidden dimension and 1024 ffn dimension as shown in the config files.
