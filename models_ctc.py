import torch
from torch import nn

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.spectr_augment import SpecAugment

from models_encoders import MidCycleConformerEncoder, CycleConformerEncoder, SeqConformerEncoder, MidSeqConformerEncoder, SharedStackMidSeqConformerEncoder


class ConformerForCTC(nn.Module):
    def __init__(self, feature_extractor, pad_token_id, config_tokenizer, config_specaug, config_model, config_architecture):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pad_token_id = pad_token_id
        self.specaug = SpecAugment(**config_specaug)
        if config_architecture=="conven":
            self.conformer_enc = ConformerEncoder(**config_model)
        elif config_architecture=="midseq":
            self.conformer_enc = MidSeqConformerEncoder(**config_model)
        else:
            print("wrong conformer type!")
        
        self.lm_head = nn.Linear(config_model.d_model, config_tokenizer.vocab_size)

    def forward(self, audio_signal, length, label=None):
        # audio_signal: [B, F, T], length: [B]
        audio_signal, length = self.feature_extractor(input_signal=audio_signal, length=length)
        if self.training:
            audio_signal = self.specaug(input_spec=audio_signal, length=length)

        # audio_signal: [B, D, T], length: [B]
        audio_signal, length = self.conformer_enc(audio_signal=audio_signal, length=length)

        # audio signal: [B, T, D]
        audio_signal = torch.transpose(audio_signal, 1, 2)
        logits = self.lm_head(audio_signal)

        # loss
        loss = None
        if label is not None:
            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = label >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = label.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    length,
                    target_lengths,
                    blank=self.pad_token_id,
                    reduction="mean",
                    zero_infinity=False
                )
        
        return {"loss": loss, "logits": logits}