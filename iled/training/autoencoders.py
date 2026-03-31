from dataclasses import dataclass
import torch.nn as nn


@dataclass
class AutoEncoderConfig:

    encoder_config: ...
    decoder_config: ...

    def make(self):
        return AutoEncoder(self.encoder_config.make(), self.decoder_config.make())


class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def transform(self, x):
        return self.encoder(x)

    def inverse_transform(self, z):
        return self.decoder(z)

    def apply_merged_func(self, batch, func_to_wrap):

        in_shape = batch.shape

        merged_batch = batch.reshape(in_shape[0] * in_shape[1], *in_shape[2:])

        out = func_to_wrap(merged_batch)

        out = out.reshape(in_shape[0], in_shape[1], *out.shape[1:])

        return out

    def batch_transform(self, batch):

        return self.apply_merged_func(batch, self.transform)

    def batch_inverse_transform(self, batch):

        return self.apply_merged_func(batch, self.inverse_transform)


@dataclass
class ScaledAutoEncoderConfig:

    encoder_config: ...
    decoder_config: ...
    scaler_config: ...

    def make(self):
        return ScaledAutoEncoder(self)


class ScaledAutoEncoder(AutoEncoder):

    def __init__(self, config):

        encoder = config.encoder_config.make()
        decoder = config.decoder_config.make()
        scaler = config.scaler_config.make()

        encoder = nn.Sequential(scaler.scaling, encoder)
        decoder = nn.Sequential(decoder, scaler.inverse_scaling)
        super().__init__(encoder, decoder)

        self.config = config
