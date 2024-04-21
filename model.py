import torch
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration, ViTModel
from transformers.modeling_outputs import BaseModelOutput


def reparameterize(mean, log_var):
    epsilon = torch.randn_like(mean).to(mean.device)
    return mean + epsilon * torch.exp(0.5 * log_var)


def kl_divergence_loss(mean_1, log_var_1, mean_2, log_var_2):
    var_1 = torch.exp(0.5 * log_var_1)
    var_2 = torch.exp(0.5 * log_var_2)
    p = Normal(mean_1, var_1)
    q = Normal(mean_2, var_2)
    return kl_divergence(p, q).mean(dim=0).sum()


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BartForMMSS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bart = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=config.language_foundation_model)
        self.vit = ViTModel.from_pretrained(
            pretrained_model_name_or_path=config.vision_foundation_model)
        self.vision_mean = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.GELU(),
            nn.Linear(in_features=768, out_features=768, bias=True)
        )
        self.vision_log_var = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.GELU(),
            nn.Linear(in_features=768, out_features=768, bias=True)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=2 * 768, out_features=1, bias=True)
        )
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pixel_values, sentence_input_ids, sentence_attention_mask,
                summarization_input_ids, summarization_attention_mask, labels):
        decoder_input_ids = shift_tokens_right(input_ids=labels, pad_token_id=self.bart.config.pad_token_id,
                                               decoder_start_token_id=self.bart.config.decoder_start_token_id)

        language_encoder_outputs = self.bart.model.encoder(
            input_ids=sentence_input_ids,
            attention_mask=sentence_attention_mask,
            return_dict=True
        )

        vision_encoder_outputs = self.vit(
            pixel_values=pixel_values
        )

        image_means = self.vision_mean(vision_encoder_outputs['pooler_output'])
        image_log_vars = self.vision_log_var(vision_encoder_outputs['pooler_output'])

        kld_loss = kl_divergence_loss(image_means, image_log_vars,
                                      torch.zeros_like(image_means), torch.ones_like(image_log_vars))

        image_embeds = reparameterize(mean=image_means, log_var=image_log_vars)

        summarization_embeds = self.bart.model.encoder(
            input_ids=summarization_input_ids,
            attention_mask=summarization_attention_mask,
            return_dict=True
        )['last_hidden_state'].min(dim=1).values

        positive = self.discriminator(torch.cat((image_embeds, summarization_embeds), dim=-1))
        positive_loss = self.bce(positive, torch.ones_like(positive))

        negative_summarization_embeds = language_encoder_outputs['last_hidden_state'].min(dim=1).values
        negative = self.discriminator(torch.cat((image_embeds, negative_summarization_embeds), dim=-1))
        negative_loss = self.bce(negative, torch.zeros_like(negative))

        last_hidden_state = torch.cat(
            tensors=(language_encoder_outputs['last_hidden_state'], image_embeds.unsqueeze(1)),
            dim=1)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None
        )

        outputs = self.bart.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            use_cache=False,
            return_dict=True
        )

        lm_logits = self.bart.lm_head(outputs['last_hidden_state'])
        lm_logits = lm_logits + self.bart.final_logits_bias.to(lm_logits.device)

        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.bart.config.vocab_size), labels.view(-1))

        total_loss = masked_lm_loss + \
                     self.config.kld_loss_weight * kld_loss + \
                     self.config.positive_loss_weight * positive_loss + \
                     self.config.negative_loss_weight * negative_loss

        return total_loss, masked_lm_loss, kld_loss, positive_loss, negative_loss

    def generate(self, pixel_values, sentence_input_ids, sentence_attention_mask):
        language_encoder_outputs = self.bart.model.encoder(
            input_ids=sentence_input_ids,
            attention_mask=sentence_attention_mask,
            return_dict=True
        )

        vision_encoder_outputs = self.vit(
            pixel_values=pixel_values
        )

        image_embeds = self.vision_mean(vision_encoder_outputs['pooler_output'])

        last_hidden_state = torch.cat(
            tensors=(language_encoder_outputs['last_hidden_state'], image_embeds.unsqueeze(1)),
            dim=1)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=None,
            attentions=None
        )

        return self.bart.generate(encoder_outputs=encoder_outputs, max_length=self.config.max_summarization_length,
                                  num_beams=self.config.num_beams)
