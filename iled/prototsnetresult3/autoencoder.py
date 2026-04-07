import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_autoencoder(model, train_loader, test_loader, device, log=print, num_epochs=300, save_path=None, patience=20):
    calc_mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    model.to(device)

    best_test_loss = float("inf")
    loss_history = []  # for printing moving average to see through cyclic LR noise

    best_weights = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # training
        model.train()

        n_examples = 0
        train_loss = 0
        total_grad_norm = 0
        num_batches = 0

        for data, _ in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            outputs, _ = model(data)
            loss = calc_mse_loss(outputs, data)
            loss.backward()

            # gradient norm — detects vanishing (<0.001) or exploding (>100) gradients
            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5
            total_grad_norm += grad_norm
            num_batches += 1

            optimizer.step()

            train_loss += loss.item() * data.size(0)
            n_examples += data.size(0)

        train_loss /= n_examples
        avg_grad_norm = total_grad_norm / max(num_batches, 1)
        loss_history.append(train_loss)
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()

            test_loss = 0
            n_examples = 0

            all_encoded = []   # for latent space statistics
            all_outputs = []
            all_inputs  = []

            for data, _ in test_loader:
                data = data.to(device)
                with torch.no_grad():
                    outputs, encoded = model(data)
                    loss = calc_mse_loss(outputs, data)

                n_examples += data.size(0)
                test_loss += loss.item() * data.size(0)

                all_encoded.append(encoded.cpu())
                all_outputs.append(outputs.cpu())
                all_inputs.append(data.cpu())

            test_loss /= n_examples
            loss_history_window = loss_history[-10:]
            moving_avg = sum(loss_history_window) / len(loss_history_window)
            best_marker = ""
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_marker = "  ← best"

            # --- Latent space diagnostics ---
            Z = torch.cat(all_encoded, dim=0)           # (N, latent, T)
            # mean activation per latent feature (collapse = all near 0)
            latent_means = Z.mean(dim=(0, 2))
            # std per latent feature (collapse = all near 0)
            latent_stds  = Z.std(dim=(0, 2))
            # dead features: std < 0.01 means that feature is doing nothing
            dead_features = (latent_stds < 0.01).sum().item()

            # --- Reconstruction R² ---
            X_in  = torch.cat(all_inputs,  dim=0).float()
            X_out = torch.cat(all_outputs, dim=0).float()
            ss_res = ((X_in - X_out) ** 2).sum().item()
            ss_tot = ((X_in - X_in.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / (ss_tot + 1e-12)

            log(
                f"epoch: {epoch:4d}/{num_epochs} | "
                f"train: {train_loss:.4f}  test: {test_loss:.4f}  "
                f"(10ep avg: {moving_avg:.4f}) | "
                f"R²: {r2:.4f} | "
                f"grad norm: {avg_grad_norm:.4f} | "
                f"latent stds: [{' '.join(f'{s:.2f}' for s in latent_stds.tolist())}] | "
                f"dead features: {dead_features}/{Z.shape[1]}"
                f"{best_marker}",
                flush=True,
            )
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                best_marker = "  ← best"
            else:
                epochs_no_improve += 10
                best_marker = ""

            if epochs_no_improve >= patience * 10:
                log(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        log(f"Restored best weights (test loss: {best_test_loss:.4f})", flush=True)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)


class PermutingConvAutoencoder(nn.Module):
    def __init__(self, num_features, latent_features, reception_percent, padding, do_max_pool=False, do_batch_norm=True):
        super(PermutingConvAutoencoder, self).__init__()
        self.do_max_pool = do_max_pool
        self.do_batch_norm = do_batch_norm

        random_state = random.getstate()
        try:
            random.seed(42)
            for _ in range(100):
                # It may happen that feature is not taken into account at all, or it's
                # taken into account by all the latent features, let's regenerate
                # permutations in such cases. 100 attempts, after that we raise an error.
                self.receive_from = []
                input_features_per_latent = max(int(reception_percent * num_features), 1)
                for _ in range(latent_features):
                    curr_receive_from = random.sample(range(num_features), input_features_per_latent)
                    curr_receive_from.sort()
                    self.receive_from.append(curr_receive_from)
                counter = Counter([item for curr_receive_from in self.receive_from for item in curr_receive_from])
                if len(counter) == num_features and all(cnt != latent_features for cnt in counter.values()):
                    break
            else:
                raise RuntimeError("Could not generate satisfying permutations, aborting")
        finally:
            random.setstate(random_state)

        self.masks = nn.Parameter(
            torch.FloatTensor([[1 if i in curr_receive_from else 0 for i in range(num_features)] for curr_receive_from in self.receive_from]),
            requires_grad=False,
        )

        self.encoder = MultiEncoder(num_features, self.masks, padding, do_max_pool=do_max_pool, do_batch_norm=do_batch_norm)

        # Decoder
        layers = []
        if do_max_pool:
            layers.append(nn.MaxUnpool1d(kernel_size=2))
        layers.append(nn.ConvTranspose1d(latent_features, 32, kernel_size=3, padding=1 if padding == "same" else 0, output_padding=0))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxUnpool1d(kernel_size=2))
        layers.append(nn.ConvTranspose1d(32, 32, kernel_size=5, padding=2 if padding == "same" else 0, output_padding=0))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxUnpool1d(kernel_size=2))
        layers.append(nn.ConvTranspose1d(32, num_features, kernel_size=7, padding=3 if padding == "same" else 0, output_padding=0))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = F.dropout(encoded, p=0.3, training=self.training)
        decoded = self.decoder(encoded)
        return decoded, encoded


class MultiEncoder(nn.Module):
    def __init__(self, num_features, masks, padding, do_max_pool=False, do_batch_norm=True):
        super(MultiEncoder, self).__init__()
        self.return_indices = do_max_pool
        self.num_branches = len(masks)

        layers = []
        layers.append(
            nn.Conv1d(
                in_channels=num_features * self.num_branches,
                out_channels=16 * self.num_branches,
                kernel_size=7,
                padding=padding,
                groups=self.num_branches,
            )
        )
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(16 * self.num_branches))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))
        layers.append(
            nn.Conv1d(
                in_channels=16 * self.num_branches, out_channels=16 * self.num_branches, kernel_size=5, padding=padding, groups=self.num_branches
            )
        )
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(16 * self.num_branches))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))
        layers.append(
            nn.Conv1d(in_channels=16 * self.num_branches, out_channels=self.num_branches, kernel_size=3, padding=padding, groups=self.num_branches)
        )
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(self.num_branches))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))

        self.encoder = nn.Sequential(*layers)

        self.masks = nn.Parameter(masks, requires_grad=False)

    def forward(self, x):
        # Apply masks
        x = x.repeat(1, self.num_branches, 1)
        x *= self.masks.view(1, -1).repeat(1, 1).unsqueeze(-1)

        # Apply encoder
        encoded = self.encoder(x)

        return encoded

    def set_return_indices(self, return_indices):
        if return_indices == self.return_indices:
            return

        self.return_indices = return_indices
        self.encoder[3 if self.do_batch_norm else 2] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[7 if self.do_batch_norm else 5] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[11 if self.do_batch_norm else 8] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)

    def set_requires_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad


class RegularConvAutoencoder(nn.Module):
    def __init__(self, num_features, latent_features, padding, do_max_pool=False, do_batch_norm=True, num_conv_filters=128):
        super(RegularConvAutoencoder, self).__init__()
        self.do_max_pool = do_max_pool
        self.return_indices = do_max_pool
        self.num_conv_filters = num_conv_filters

        self.encoder = RegularConvEncoder(
            num_features, latent_features, padding, do_max_pool=do_max_pool, do_batch_norm=do_batch_norm, num_conv_filters=num_conv_filters
        )

        layers = []
        if do_max_pool:
            layers.append(nn.MaxUnpool1d(kernel_size=2))
        layers.append(nn.ConvTranspose1d(latent_features, num_conv_filters, kernel_size=3, padding=1 if padding == "same" else 0, output_padding=0))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxUnpool1d(kernel_size=2))
        layers.append(nn.ConvTranspose1d(num_conv_filters, num_conv_filters, kernel_size=5, padding=2 if padding == "same" else 0, output_padding=0))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxUnpool1d(kernel_size=2))
        layers.append(nn.ConvTranspose1d(num_conv_filters, num_features, kernel_size=7, padding=3 if padding == "same" else 0, output_padding=0))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        if self.return_indices:
            encoded, indices, sizes = self.encoder(x)
            encoded = F.dropout(encoded, p=0.5, training=self.training)
            indices = indices[::-1]
            sizes = sizes[::-1]
            decoded = encoded
            for i, layer in enumerate(self.decoder):
                if isinstance(layer, nn.MaxUnpool1d):
                    decoded = layer(decoded, indices[i // 3], output_size=sizes[i // 3])
                else:
                    decoded = layer(decoded)
        else:
            encoded = self.encoder(x)
            encoded = F.dropout(encoded, p=0.3, training=self.training)
            decoded = self.decoder(encoded)
        return decoded, encoded


class RegularConvEncoder(nn.Module):
    def __init__(self, num_features, latent_features, padding, do_max_pool=False, do_batch_norm=True, num_conv_filters=32):
        super(RegularConvEncoder, self).__init__()
        self.return_indices = do_max_pool
        self.do_batch_norm = do_batch_norm
        self.num_conv_filters = num_conv_filters
        self.kernel_sizes = [7, 5, 3]

        layers = []

        layers.append(nn.Conv1d(in_channels=num_features, out_channels=num_conv_filters, kernel_size=self.kernel_sizes[0], padding=padding))
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(num_conv_filters))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))
        layers.append(nn.Conv1d(in_channels=num_conv_filters, out_channels=num_conv_filters, kernel_size=self.kernel_sizes[1], padding=padding))
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(num_conv_filters))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))
        layers.append(nn.Conv1d(in_channels=num_conv_filters, out_channels=latent_features, kernel_size=self.kernel_sizes[2], padding=padding))
        if do_batch_norm:
            layers.append(nn.BatchNorm1d(latent_features))
        layers.append(nn.ReLU())
        if do_max_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, return_indices=do_max_pool))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        if self.return_indices:
            indices = []
            sizes = []
            for layer in self.encoder:
                if isinstance(layer, nn.MaxPool1d):
                    sizes.append(x.size())
                    x, idx = layer(x)
                    indices.append(idx)
                else:
                    x = layer(x)
            return x, indices, sizes
        else:
            return self.encoder(x)

    def set_return_indices(self, return_indices):
        if return_indices == self.return_indices:
            return

        self.return_indices = return_indices
        self.encoder[3 if self.do_batch_norm else 2] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[7 if self.do_batch_norm else 5] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)
        self.encoder[11 if self.do_batch_norm else 8] = nn.MaxPool1d(kernel_size=2, return_indices=return_indices)

    def set_requires_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad