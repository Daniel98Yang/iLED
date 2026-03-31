#!/usr/bin/env python

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder

from torcheval.metrics.functional import multiclass_confusion_matrix

from artificial_protos_datasets import ArtificialProtos
from autoencoder import PermutingConvAutoencoder, RegularConvAutoencoder, train_autoencoder
from datasets_utils import TSCDataset, ds_load, transform_ts_data
from model import ProtoTSNet
from train import EpochType, ProtoTSCoeffs, create_logger, train_prototsnet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# write your own function to load your dataset...
def custom_dataset():

    train_file = "/home/sse-3w5wqn3/ProtoTSNet/datasets/heli/train.arff"
    test_file  = "/home/sse-3w5wqn3/ProtoTSNet/datasets/heli/test.arff"

    T = 200   # timesteps
    D = 314   # channels after filtering

    label_encoder = LabelEncoder()

    def arff_to_numpy(file_path):
        data, _ = loadarff(file_path)
        X, y = [], []
        for row in data:
            try:
                x, label = row
            except ValueError:
                row = row.tolist()
                x = np.array(row[:-1]).reshape(1, -1)
                label = row[-1]
            x = np.array(x.tolist(), dtype="float32")
            X.append(x)
            y.append(label)
        return np.nan_to_num(np.stack(X, axis=0), 0), np.stack(y, axis=0)

    X_train, y_train = arff_to_numpy(train_file)
    X_test,  y_test  = arff_to_numpy(test_file)

    # reshape from (N, T*D) back to (N, D, T) — channels first for ProtoTSNet
    X_train = X_train.reshape(-1, T, D).transpose(0, 2, 1)  # (N, D, T)
    X_train += 0.001 * np.random.randn(*X_train.shape)
    X_test  = X_test.reshape(-1, T, D).transpose(0, 2, 1)

    y_train = label_encoder.fit_transform(y_train)
    y_test  = label_encoder.transform(y_test)

    # scaler = StandardScaler()
    # X_train = transform_ts_data(X_train, scaler, scale_separately=False, fit=True)
    # X_test  = transform_ts_data(X_test,  scaler, scale_separately=False, fit=False)

    
    return TSCDataset(X_train, y_train), TSCDataset(X_test, y_test)

# ... like this:
# and update the code below to use it (look for pm_dataset usage)
def pm_dataset(use_pm=True):
    # replicates the transformations described in the LAXCAT paper (to the best of our understanding)
    df = pd.read_csv("./datasets/PRSA_data_2010.1.1-2014.12.31.csv")
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df.set_index("datetime", inplace=True)
    df = df.dropna()

    day_counts = df.groupby(df.index.date).size()
    full_days = day_counts[day_counts == 24].index
    df = df[pd.Index(df.index.date).isin(full_days)]
    wd = {
        "cv": 0,
        "NW": 1,
        "NE": 2,
        "SE": 3,
    }
    df["cbwd"] = df["cbwd"].map(wd)
    dates = df.index.date
    unique_dates = set(dates)

    xs = []
    ys = []
    for date in unique_dates:
        next_day = date + pd.Timedelta(days=1)
        if next_day in unique_dates and next_day.weekday() < 5:
            next_day_8am = pd.Timestamp(next_day) + pd.Timedelta(hours=8)
            data = df.loc[df.index.date == date, ["pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]].to_numpy()
            if not use_pm:
                data = data[:, 1:]
            pm_value = df.loc[next_day_8am, "pm2.5"]
            xs.append(data)
            # quantize pm2.5 according to US EPA standards
            ys.append(
                0 if pm_value < 12 else 1 if pm_value < 35.4 else 2 if pm_value < 55.4 else 3 if pm_value < 150.4 else 4 if pm_value < 250.4 else 5
            )

    X = np.array(xs, dtype=np.float32).transpose(0, 2, 1)
    y = np.array(ys)
    X_train = X[: int(0.7 * X.shape[0])]
    y_train = y[: int(0.7 * X.shape[0])]
    X_test = X[int(0.7 * X.shape[0]) :]
    y_test = y[int(0.7 * X.shape[0]) :]
    train_ds = TSCDataset(X_train, y_train)
    test_ds = TSCDataset(X_test, y_test)
    return train_ds, test_ds


def artificial_dataset():
    log(f"Preparing artificial dataset...", flush=True, display=True)
    train_art = ArtificialProtos(1000, feature_noise_power=0.1, randomize_right_side=True)
    train_ds = TSCDataset(train_art.X, train_art.y)
    test_art = ArtificialProtos(100, feature_noise_power=0.1, randomize_right_side=True)
    test_ds = TSCDataset(test_art.X, test_art.y)
    return train_ds, test_ds


def experiment_setup(experiment_subpath):
    # create subdirectory for the experiment and copy files to save state of the code
    experiment_dir = Path.cwd() / "experiments" / experiment_subpath
    os.makedirs(experiment_dir, exist_ok=True)

    shutil.copy(src=Path.cwd() / "autoencoder.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "datasets_utils.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "main.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "model.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "push.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "receptive_field.py", dst=experiment_dir)
    shutil.copy(src=Path.cwd() / "train.py", dst=experiment_dir)

    return experiment_dir


def validate_proto_len(proto_len):
    flen = float(proto_len)
    if flen < 0 or flen > 1:
        parser.error("proto_len must be in range (0, 1]")
    return flen

def augment_timeseries(X, noise_level=0.15, max_shift=10):
    """
    X: shape (N, C, T)
    noise_level: relative noise (fraction of signal std)
    max_shift: max temporal shift (in timesteps)
    """

    # ---- 1. Compute signal scale per sample/channel ----
    # std over time axis
    signal_std = np.std(X, axis=-1, keepdims=True)  # shape (N, C, 1)

    # avoid zero std
    signal_std = np.maximum(signal_std, 1e-6)

    # ---- 2. Add scaled noise ----
    noise = np.random.normal(0, 1, size=X.shape).astype(X.dtype)
    noise = noise * signal_std * noise_level
    X_noisy = X + noise

    # ---- 3. Temporal shift (per sample) ----
    X_aug = np.empty_like(X_noisy)

    for i in range(X.shape[0]):
        shift = int(np.random.normal(0, max_shift/2))
        X_aug[i] = np.roll(X_noisy[i], shift, axis=-1)

    return X_aug.astype(X.dtype)




# dataset in arff format should be put in the 'datasets/' directory (downloaded from timeseriesclassification.com)
DATASETS_PATH = Path("datasets")

parser = argparse.ArgumentParser(description="Run experiment with specified dataset and experiment directory.")
parser.add_argument("--uea_dataset", type=str, help="Use this if you want to run experiment on UEA dataset under datasets/ directory", required=False)
parser.add_argument("--artificial_dataset", action="store_true", help="Run experiments on artificial dataset", required=False)
parser.add_argument("--experiment_name", type=str, required=True, help="Directory to save experiment results")
parser.add_argument("--no_permuting_encoder", action="store_true", help="Do not use permuting encoder", required=False)
parser.add_argument("--no_encoder_pretraining", action="store_true", help="Do not pretrain encoder before ProtoTSNet training", required=False)
parser.add_argument("--pretraining_epochs", type=int, help="Number of encoder pretraining epochs", required=False, default=50)
parser.add_argument("--num_warm_epochs", type=int, help="Number of warm-up epochs", required=False, default=None)
parser.add_argument("--push_start_epoch", type=int, help="Epoch to start pushing prototypes", required=False, default=150)
parser.add_argument("--push_epochs_interval", type=int, help="Interval between pushing prototypes", required=False, default=30)
parser.add_argument("--last_layer_epochs", type=int, help="Number of epochs to train last layer", required=False, default=20)
parser.add_argument("--epochs", type=int, help="Number of epochs to train", required=False, default=200)
parser.add_argument("--proto_features", type=int, help="Number of latent features", required=False, default=32)
parser.add_argument("--proto_len", type=validate_proto_len, help="Prototype length (0-1 range, fraction of series length)", required=False, default=None)
parser.add_argument("--protos_per_class", type=int, help="Number of prototypes for each class", required=False, default=10)
parser.add_argument("--reception", type=float, help="Fraction of significant features", required=False, default=None)
parser.add_argument("--l1_addon_coeff", type=float, help="L1 regularization coefficient for feature importance layer", required=False, default=1e-3)
parser.add_argument("--l1_coeff", type=float, help="L1 regularization coefficient", required=False, default=5e-3)
parser.add_argument("--clst_coeff", type=float, help="Cluster separation coefficient", required=False, default=0.8)
parser.add_argument("--sep_coeff", type=float, help="Separation coefficient", required=False, default=-0.08)
parser.add_argument("--param_selection", action="store_true", help="Run hyperparameter selection", required=False, default=False)
parser.add_argument("--verbose", action="store_true", help="Verbose output", required=False, default=False)
parser.add_argument("--skip_scaling", action="store_true", help="Skip scaling of the dataset", required=False, default=False)
parser.add_argument("--early_stop_patience", type=int, default=50)
parser.add_argument("--early_stop_min_epochs", type=int, default=100)
parser.add_argument("--early_stop_acc_threshold", type=float, default=0.9)

args = parser.parse_args()

ds_name = args.uea_dataset

if ds_name is not None:
    # read best_params.csv
    try:
        best_params = pd.read_csv("best_params.csv", index_col=0)
    except Exception as e:
        best_params = None
else:
    best_params = None

if best_params is None:
    if not args.proto_len or not args.reception:
        parser.error("Prototype length and reception must be provided for non-UEA datasets, or when best_params.csv is missing")

experiment_name = f"{args.experiment_name}/{ds_name}" if ds_name is not None else args.experiment_name
experiment_dir = experiment_setup(experiment_name)
log, logclose = create_logger(experiment_dir / "log.txt", display=args.verbose)

if args.skip_scaling:
    skip_scaling = True
elif best_params is not None:
    skip_scaling = best_params.loc[ds_name, "skip_scaling"] == "T"
else:
    skip_scaling = False

if ds_name is not None:
    log(f"Loading dataset {ds_name}...", flush=True, display=True)
    train_ds, test_ds = custom_dataset()
elif args.artificial_dataset:
    train_ds, test_ds = custom_dataset()
else:
    log(f"Running on custom dataset", flush=True, display=True)
    train_ds, test_ds = custom_dataset()

if not skip_scaling:
    log(f"Scaling dataset...", flush=True, display=True)
    scaler = StandardScaler()
    train_ds.X = transform_ts_data(train_ds.X, scaler, scale_separately=True)
    test_ds.X = transform_ts_data(test_ds.X, scaler, scale_separately=True, fit=False)
    # apply ONLY to test data
    test_ds.X = augment_timeseries(test_ds.X, noise_level=0.05, max_shift=10)
    scaler_dir = Path(experiment_dir) / "scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = scaler_dir / f"{experiment_name}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"AIEEEEEEEE")
else:
    log(f"Skipping scaling of the dataset", flush=True, display=True)

# hyperparameters

# number of prototypes will equal 'protos_per_class * number of classes'
protos_per_class = args.protos_per_class

# prototype length (number of time steps) - it is latent space length, so due to receptive field in the input space it is longer
proto_len = float(best_params.loc[ds_name, "proto_len"]) if args.proto_len is None else args.proto_len

# number of latent features (dimensions) that input is encoded to
proto_features = args.proto_features

# estimate for the fraction of significant features, better to underestimate than overestimate
reception = float(best_params.loc[ds_name, "reception"]) if args.reception is None else args.reception
permuting_encoder = not args.no_permuting_encoder
encoder_pretraining = not args.no_encoder_pretraining

# number of epochs during which encoder weights are frozen, value >0 only makes sense if encoder is pretrained
num_warm_epochs = args.num_warm_epochs if args.num_warm_epochs is not None else 50 if encoder_pretraining else 0

# when to start pushing prototypes onto the input data
push_start_epoch = args.push_start_epoch

# which epochs to push prototypes on
push_epochs = range(push_start_epoch, 1000, args.push_epochs_interval)

# how many epochs to train the last layer (prototypes <-> class mapping)
num_last_layer_epochs = args.last_layer_epochs

# overall number of epochs (PUSH + last layer "epochs" count as one epoch here), set it so that the training ends with PUSH
epochs = args.epochs

# how much each element contributes to the loss, l1 is last layer l1 regularization, l1_addon is regularization of feature importance layer
coeffs = ProtoTSCoeffs(crs_ent=1, l1_addon=args.l1_addon_coeff, l1=args.l1_coeff, clst=args.clst_coeff, sep=args.sep_coeff)

# retrieve details of the dataset
num_classes = len(np.unique(train_ds.y))
num_features = train_ds.X.shape[1]
ts_len = train_ds.X.shape[2]

if isinstance(proto_len, float):
    proto_len = int(ts_len * proto_len)

if num_features == 1:
    # this is an univariate dataset, no point in using permuting encoder
    permuting_encoder = False


def setup_and_run_experiment(experiment_name, experiment_dir, log, train_ds, test_ds, reception, proto_len):
    try:
        train_batch_size = 32
        # reduce in case dataset is small
        while train_batch_size > len(train_ds.X) / 4:
            train_batch_size //= 2
        test_batch_size = 128

        do_batch_norm = True
        if train_batch_size < 8:
            do_batch_norm = False

        whole_training_start = time.time()

        if permuting_encoder:
            autoencoder = PermutingConvAutoencoder(
                num_features=num_features,
                latent_features=proto_features,
                reception_percent=reception,
                padding="same",
                do_max_pool=False,
                do_batch_norm=do_batch_norm,
            )
        else:
            autoencoder = RegularConvAutoencoder(
                num_features=num_features, latent_features=proto_features, padding="same", do_max_pool=False, do_batch_norm=do_batch_norm
            )
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
        ae_save_path = Path(experiment_dir) / "autoencoder.pt"
        
        print(f"Train batches: {len(train_loader)}")
        print(f"X shape: {train_ds.X.shape}")


        if encoder_pretraining:
            if ae_save_path.exists():
                log(f"Loading pretrained autoencoder from {ae_save_path}", flush=True, display=True)
                autoencoder.load_state_dict(torch.load(ae_save_path))
            else:
                log(f"Training encoder", flush=True, display=True)
                train_autoencoder(autoencoder, train_loader, None,
                                  device=device, log=log,
                                  num_epochs=args.pretraining_epochs,
                                  save_path=ae_save_path)
                torch.save(autoencoder.state_dict(), experiment_dir / 'autoencoder_pretrained.pth')

            
        autoencoder.encoder.set_return_indices(False)

        ptsnet = ProtoTSNet(
            cnn_base=autoencoder.encoder,
            num_features=num_features,
            ts_sample_len=ts_len,
            proto_num=protos_per_class * num_classes,
            latent_features=proto_features,
            proto_len_latent=proto_len,
            num_classes=num_classes,
            init_encoder_weights=not encoder_pretraining,
            prototype_activation_function="log",
        )

        def lr_sched_setup(optimizer, epoch_type):
            if epoch_type == EpochType.JOINT:
                return torch.optim.lr_scheduler.CyclicLR(
                    optimizer, base_lr=1e-5, max_lr=5e-4, step_size_up=10, step_size_down=20, mode="exp_range", gamma=0.99, cycle_momentum=False
                )
            elif epoch_type == EpochType.LAST_LAYER:
                return torch.optim.lr_scheduler.CyclicLR(
                    optimizer, base_lr=1e-5, max_lr=5e-4, step_size_up=15, step_size_down=25, mode="exp_range", gamma=0.99, cycle_momentum=False
                )
            return None

        log(f"Training ProtoTSNet", flush=True, display=True)
        trainer = train_prototsnet(
            ptsnet=ptsnet,
            experiment_dir=experiment_dir,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            num_warm_epochs=num_warm_epochs,
            push_start_epoch=push_start_epoch,
            push_epochs=push_epochs,
            num_last_layer_epochs=num_last_layer_epochs,
            num_epochs=epochs,
            coeffs=coeffs,
            lr_sched_setup=lr_sched_setup,
            log=log,
            add_params_to_log={
                "encoder_pretraining": encoder_pretraining,
                "permuting_encoder": permuting_encoder,
                "reception": reception,
                "proto_len_fraction": f"{proto_len / ts_len:.3f}",
            },
        )

        accu_test = trainer.latest_stat("accu_test")
        total_time = trainer.latest_stat("total_time")
        log(f"Last epoch test accu: {accu_test*100:.2f}%", display=True)
        with open(experiment_dir / "results.json", "w") as f:
            json.dump({"accu_test": accu_test, "time": total_time}, f, indent=4)

        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size)
        confusion_matrix = torch.zeros(ptsnet.num_classes, ptsnet.num_classes)
        for i, (image, label) in enumerate(test_loader):
            output, _ = ptsnet(image.to(device))
            confusion_matrix += multiclass_confusion_matrix(output.to("cpu"), label, num_classes=output.shape[1])
        np.savetxt(experiment_dir / "confusion_matrix.txt", confusion_matrix.numpy(), fmt="%4d")

        whole_training_end = time.time()
        log(f"Done in {trainer.curr_epoch - 1} epochs, {whole_training_end - whole_training_start:.2f}s", display=True)
    except Exception as e:
        log(f"Exception ocurred for experiment {experiment_name}: {e}", display=True)
        tb_str = traceback.format_tb(e.__traceback__)
        log("\n".join(tb_str), display=True)
        raise


if not args.param_selection:
    setup_and_run_experiment(experiment_name, experiment_dir, log, train_ds, test_ds, reception, proto_len)
else:
    log(f"Running hyperparameter selection for {ds_name if ds_name else 'custom dataset'}...", flush=True, display=True)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if ds_name == "StandWalkJump":
        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    for reception in [0.25, 0.5, 0.75, 0.9]:
        for denom in [100, 10, 5, 2, 1]:
            log(f"Running for reception {reception} and proto_len {1/denom}", flush=True, display=True)
            for fold_idx, (train_ind, test_ind) in enumerate(kfold.split(train_ds.X, train_ds.y)):
                proto_len = int(ts_len // denom)
                if proto_len == 0:
                    continue

                fold_train_ds = TSCDataset(train_ds.X[train_ind], train_ds.y[train_ind])
                fold_test_ds = TSCDataset(train_ds.X[test_ind], train_ds.y[test_ind])

                fold_experiment_name = f"{experiment_name}/reception-{reception}/proto_len-{1/denom}/fold-{fold_idx}"
                fold_experiment_dir = experiment_setup(fold_experiment_name)
                fold_log, fold_logclose = create_logger(fold_experiment_dir / "log.txt", display=args.verbose)

                curr_link_path = experiment_dir / "curr_log.txt"
                if os.path.islink(curr_link_path):
                    os.unlink(curr_link_path)
                os.symlink(fold_experiment_dir / "log.txt", curr_link_path)

                setup_and_run_experiment(fold_experiment_name, fold_experiment_dir, fold_log, fold_train_ds, fold_test_ds, reception, proto_len)

                fold_logclose()
                os.unlink(curr_link_path)
