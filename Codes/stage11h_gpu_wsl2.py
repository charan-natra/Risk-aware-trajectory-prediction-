# ================================================================
# STAGE 11H
# INTERACTION-AWARE TRAJECTORY PREDICTION
# WITH OVERLAP / FEEDBACK REFINEMENT
# ================================================================
#
# Input:
#   X_motion_10hz_train.npy
#   X_context_10hz_train.npy
#   Y_10hz_train.npy
#
#   X_motion_10hz_val.npy
#   X_context_10hz_val.npy
#   Y_10hz_val.npy
#
#   X_motion_10hz_test.npy
#   X_context_10hz_test.npy
#   Y_10hz_test.npy
#
# Model:
#
#   Target Motion
#        |
#        v
#   Motion Encoder
#        |
#        v
#   Initial Trajectory
#        |
#        +----------------------+
#                               |
#   Neighbor History            |
#        |                      |
#        v                      |
#   Neighbor Encoder            |
#        |                      |
#        v                      |
#   Interaction Attention       |
#        |                      |
#        v                      |
#   Interaction Representation  |
#        |                      |
#        +----> Feedback -------+
#                               |
#                               v
#                       Refined Trajectory
#
# IMPORTANT:
#   No future neighbor trajectory is used.
#   Only historical interaction information is used.
#
# ================================================================


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# ================================================================
# GPU CONFIGURATION
# IMPORTANT:
# - Native Windows TensorFlow 2.16 does not provide CUDA GPU support.
# - Run this script in WSL2 Ubuntu with TensorFlow's Linux CUDA package.
# - Fail early if no GPU is visible, rather than silently training on CPU.
# ================================================================

physical_gpus = tf.config.list_physical_devices("GPU")
print("\nTensorFlow version:", tf.__version__)
print("Visible GPUs:", physical_gpus)

if not physical_gpus:
    raise RuntimeError(
        "No TensorFlow GPU detected. This script is intended to run in "
        "WSL2 Ubuntu with a working NVIDIA GPU and Linux TensorFlow CUDA "
        "dependencies. Do not run it in native Windows with TensorFlow 2.16."
    )

for gpu in physical_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("Could not enable memory growth:", exc)

print("GPU memory growth enabled.")
print("Training device:", physical_gpus[0].name)


from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    BatchNormalization,
    Bidirectional,
    LSTM,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    Concatenate,
    Dropout,
    TimeDistributed,
    GlobalAveragePooling1D,
    Reshape,
    RepeatVector,
    Multiply,
    Softmax
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.metrics import (
    RootMeanSquaredError
)


# ================================================================
# 1. CONFIGURATION
# ================================================================

# Windows project folder, accessed from WSL2 at /mnt/c/...
if os.name == "nt":
    BASE_DIR = r"C:\Users\natra\Final year project"
else:
    BASE_DIR = "/mnt/c/Users/natra/Final year project"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------
# Reproducibility
# ------------------------------------------------

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------
# Temporal configuration
# ------------------------------------------------

HISTORY_STEPS = 8
FUTURE_STEPS = 12

SAMPLING_FREQUENCY_HZ = 10
DT_SECONDS = 1.0 / SAMPLING_FREQUENCY_HZ


# ------------------------------------------------
# Feature configuration
# ------------------------------------------------

MOTION_FEATURES = 9

# Context:
#
# 0 relative_x
# 1 relative_y
# 2 relative_vx
# 3 relative_vy
# 4 relative_speed
# 5 distance

CONTEXT_FEATURES = 6

MAX_NEIGHBORS = 5


# ------------------------------------------------
# Training configuration
# ------------------------------------------------

EPOCHS = 35
BATCH_SIZE = 64

LEARNING_RATE = 0.0003


# ================================================================
# 2. PRINT CONFIGURATION
# ================================================================

print("=" * 70)
print("STAGE 11H - INTERACTION + FEEDBACK TRAJECTORY MODEL")
print("=" * 70)

print("\nConfiguration:")

print(
    f"Sampling frequency : "
    f"{SAMPLING_FREQUENCY_HZ} Hz"
)

print(
    f"Timestep           : "
    f"{DT_SECONDS:.3f} sec"
)

print(
    f"History steps      : "
    f"{HISTORY_STEPS}"
)

print(
    f"Future steps       : "
    f"{FUTURE_STEPS}"
)

print(
    f"Motion features    : "
    f"{MOTION_FEATURES}"
)

print(
    f"Context features   : "
    f"{CONTEXT_FEATURES}"
)

print(
    f"Maximum neighbors  : "
    f"{MAX_NEIGHBORS}"
)

print(
    f"Batch size         : "
    f"{BATCH_SIZE}"
)

print(
    f"Maximum epochs     : "
    f"{EPOCHS}"
)


# ================================================================
# 3. LOAD DATA
# ================================================================

print("\n" + "=" * 70)
print("LOADING 10-HZ DATA")
print("=" * 70)


def load_split(split):

    motion_path = os.path.join(
        OUTPUT_DIR,
        f"X_motion_10hz_{split}.npy"
    )

    context_path = os.path.join(
        OUTPUT_DIR,
        f"X_context_10hz_{split}.npy"
    )

    target_path = os.path.join(
        OUTPUT_DIR,
        f"Y_10hz_{split}.npy"
    )

    for path in [
        motion_path,
        context_path,
        target_path
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"File not found:\n{path}"
            )

    # Float32 is sufficient for trajectory training and uses less RAM/VRAM.
    motion = np.load(motion_path).astype(np.float32, copy=False)
    context = np.load(context_path).astype(np.float32, copy=False)
    target = np.load(target_path).astype(np.float32, copy=False)

    return motion, context, target


X_train, C_train, Y_train = load_split(
    "train"
)

X_val, C_val, Y_val = load_split(
    "val"
)

X_test, C_test, Y_test = load_split(
    "test"
)


# ================================================================
# 4. PRINT SHAPES
# ================================================================

print("\nTRAIN")
print(
    "Motion:",
    X_train.shape
)
print(
    "Context:",
    C_train.shape
)
print(
    "Target:",
    Y_train.shape
)


print("\nVAL")
print(
    "Motion:",
    X_val.shape
)
print(
    "Context:",
    C_val.shape
)
print(
    "Target:",
    Y_val.shape
)


print("\nTEST")
print(
    "Motion:",
    X_test.shape
)
print(
    "Context:",
    C_test.shape
)
print(
    "Target:",
    Y_test.shape
)


# ================================================================
# 5. CHECK DATA
# ================================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)


arrays = {

    "X_train": X_train,
    "C_train": C_train,
    "Y_train": Y_train,

    "X_val": X_val,
    "C_val": C_val,
    "Y_val": Y_val,

    "X_test": X_test,
    "C_test": C_test,
    "Y_test": Y_test
}


for name, array in arrays.items():

    nan_count = np.isnan(array).sum()
    inf_count = np.isinf(array).sum()

    print(
        f"{name:10s} | "
        f"NaNs: {nan_count:8d} | "
        f"Infs: {inf_count:8d}"
    )

    if nan_count > 0:

        raise ValueError(
            f"{name} contains NaN values."
        )

    if inf_count > 0:

        raise ValueError(
            f"{name} contains infinite values."
        )


# ================================================================
# 6. NORMALIZATION
# ================================================================
#
# Motion scaler:
#   fitted only on TRAIN.
#
# Context scaler:
#   fitted only on VALID neighbor entries.
#
# IMPORTANT:
#   Distance is also normalized for neural-network training.
#
# The model learns interaction relationships from the normalized
# historical context.
#
# ================================================================

print("\n" + "=" * 70)
print("NORMALIZATION")
print("=" * 70)


# ------------------------------------------------
# Motion
# ------------------------------------------------

motion_scaler = StandardScaler()

motion_scaler.fit(
    X_train.reshape(
        -1,
        MOTION_FEATURES
    )
)


X_train_scaled = motion_scaler.transform(
    X_train.reshape(
        -1,
        MOTION_FEATURES
    )
).reshape(
    X_train.shape
)


X_val_scaled = motion_scaler.transform(
    X_val.reshape(
        -1,
        MOTION_FEATURES
    )
).reshape(
    X_val.shape
)


X_test_scaled = motion_scaler.transform(
    X_test.reshape(
        -1,
        MOTION_FEATURES
    )
).reshape(
    X_test.shape
)


# ------------------------------------------------
# Context
# ------------------------------------------------

context_scaler = StandardScaler()


# Valid neighbor entries are identified by distance > 0.
#
# Padded entries have all six features equal to zero.

valid_context_train = (
    C_train.reshape(
        -1,
        CONTEXT_FEATURES
    )
)

valid_mask = (
    np.linalg.norm(
        valid_context_train,
        axis=1
    ) > 1e-8
)


if valid_mask.sum() == 0:

    raise ValueError(
        "No valid context entries found."
    )


context_scaler.fit(
    valid_context_train[
        valid_mask
    ]
)


def scale_context(
    context
):

    original_shape = context.shape

    flattened = context.reshape(
        -1,
        CONTEXT_FEATURES
    )

    scaled = np.zeros_like(
        flattened,
        dtype=np.float32
    )

    valid = (
        np.linalg.norm(
            flattened,
            axis=1
        ) > 1e-8
    )

    scaled[valid] = context_scaler.transform(
        flattened[valid]
    )

    return scaled.reshape(original_shape).astype(np.float32, copy=False)


C_train_scaled = scale_context(
    C_train
)

C_val_scaled = scale_context(
    C_val
)

C_test_scaled = scale_context(
    C_test
)


# ------------------------------------------------
# Target
# ------------------------------------------------

target_scaler = StandardScaler()

target_scaler.fit(
    Y_train.reshape(
        -1,
        2
    )
)


Y_train_scaled = target_scaler.transform(
    Y_train.reshape(
        -1,
        2
    )
).reshape(
    Y_train.shape
)


Y_val_scaled = target_scaler.transform(
    Y_val.reshape(
        -1,
        2
    )
).reshape(
    Y_val.shape
)


Y_test_scaled = target_scaler.transform(
    Y_test.reshape(
        -1,
        2
    )
).reshape(
    Y_test.shape
)


print(
    "Normalization complete."
)


# ================================================================
# 7. SAVE SCALERS
# ================================================================

normalization_path = os.path.join(
    OUTPUT_DIR,
    "stage11h_interaction_feedback_normalization.npz"
)


np.savez(

    normalization_path,

    motion_mean=motion_scaler.mean_,
    motion_scale=motion_scaler.scale_,

    context_mean=context_scaler.mean_,
    context_scale=context_scaler.scale_,

    target_mean=target_scaler.mean_,
    target_scale=target_scaler.scale_
)


print(
    "\nNormalization parameters saved:"
)

print(
    normalization_path
)


# ================================================================
# 8. BUILD INTERACTION-AWARE MODEL
# ================================================================

print("\n" + "=" * 70)
print("BUILDING INTERACTION-AWARE MODEL")
print("=" * 70)


# ================================================================
# TARGET MOTION INPUT
# ================================================================

motion_input = Input(

    shape=(
        HISTORY_STEPS,
        MOTION_FEATURES
    ),

    name="target_motion"
)


# ================================================================
# NEIGHBOR CONTEXT INPUT
# ================================================================

context_input = Input(

    shape=(
        HISTORY_STEPS,
        MAX_NEIGHBORS,
        CONTEXT_FEATURES
    ),

    name="neighbor_context"
)


# ================================================================
# TARGET MOTION ENCODER
# ================================================================

motion = Conv1D(

    64,

    kernel_size=3,

    padding="same",

    activation="relu",

    name="motion_conv_64"

)(motion_input)


motion = BatchNormalization(
    name="motion_bn_64"
)(motion)


motion = Conv1D(

    128,

    kernel_size=3,

    padding="same",

    activation="relu",

    name="motion_conv_128"

)(motion)


motion = BatchNormalization(
    name="motion_bn_128"
)(motion)


motion = Bidirectional(

    LSTM(
        128,
        return_sequences=True
    ),

    name="motion_bilstm"

)(motion)


# ================================================================
# TARGET SELF-ATTENTION
# ================================================================

motion_attention = MultiHeadAttention(

    num_heads=4,

    key_dim=32,

    dropout=0.10,

    name="motion_self_attention"

)(
    motion,
    motion
)


motion = Add(
    name="motion_attention_residual"
)([
    motion,
    motion_attention
])


motion = LayerNormalization(
    name="motion_attention_norm"
)(motion)


# ================================================================
# INITIAL TRAJECTORY REPRESENTATION
# ================================================================

motion_global = GlobalAveragePooling1D(
    name="motion_global_pool"
)(motion)


motion_global = Dense(
    128,
    activation="relu",
    name="motion_embedding"
)(motion_global)


motion_global = Dropout(
    0.10
)(motion_global)


# ================================================================
# INITIAL TRAJECTORY
# ================================================================
#
# This is an intermediate trajectory representation.
#
# It is NOT the final output.
#
# Interaction information will later refine it.
#
# ================================================================

initial_decoder = RepeatVector(
    FUTURE_STEPS,
    name="initial_repeat"
)(motion_global)


initial_decoder = LSTM(
    128,
    return_sequences=True,
    name="initial_trajectory_decoder"
)(initial_decoder)


initial_decoder = Dropout(
    0.10
)(initial_decoder)


initial_trajectory = TimeDistributed(

    Dense(
        2,
        name="initial_xy"
    ),

    name="initial_prediction"

)(initial_decoder)


# ================================================================
# NEIGHBOR ENCODER
# ================================================================
#
# Input:
#
#   batch
#   × 8 history
#   × 5 neighbors
#   × 6 features
#
# We first flatten the neighbor feature dimension:
#
#   5 × 6 = 30
#
# The temporal dimension remains 8.
#
# ================================================================

context = Reshape(

    (
        HISTORY_STEPS,
        MAX_NEIGHBORS * CONTEXT_FEATURES
    ),

    name="flatten_neighbor_features"

)(context_input)


context = Conv1D(

    64,

    kernel_size=3,

    padding="same",

    activation="relu",

    name="context_conv_64"

)(context)


context = BatchNormalization(
    name="context_bn"
)(context)


context = Bidirectional(

    LSTM(
        64,
        return_sequences=True
    ),

    name="context_temporal_encoder"

)(context)


# ================================================================
# CONTEXT SELF-ATTENTION
# ================================================================

context_attention = MultiHeadAttention(

    num_heads=4,

    key_dim=32,

    dropout=0.10,

    name="context_self_attention"

)(
    context,
    context
)


context = Add(
    name="context_attention_residual"
)([
    context,
    context_attention
])


context = LayerNormalization(
    name="context_attention_norm"
)(context)


# ================================================================
# GLOBAL INTERACTION REPRESENTATION
# ================================================================

interaction_global = GlobalAveragePooling1D(
    name="interaction_global_pool"
)(context)


interaction_global = Dense(
    128,
    activation="relu",
    name="interaction_embedding"
)(interaction_global)


interaction_global = Dropout(
    0.10
)(interaction_global)


# ================================================================
# TARGET-CONDITIONED INTERACTION
# ================================================================
#
# Combine target embedding and interaction embedding.
#
# This represents:
#
#   "What does the surrounding traffic mean
#    for this particular target vehicle?"
#
# ================================================================

interaction_fusion = Concatenate(
    name="target_conditioned_interaction"
)([
    motion_global,
    interaction_global
])


interaction_fusion = Dense(
    256,
    activation="relu",
    name="interaction_fusion_dense"
)(interaction_fusion)


interaction_fusion = Dropout(
    0.10
)(interaction_fusion)


interaction_fusion = Dense(
    128,
    activation="relu",
    name="interaction_fusion_embedding"
)(interaction_fusion)


# ================================================================
# FEEDBACK VECTOR
# ================================================================
#
# The interaction representation produces a feedback signal
# which modifies the initial trajectory.
#
# ================================================================

feedback = Dense(
    128,
    activation="sigmoid",
    name="interaction_feedback_gate"
)(interaction_fusion)


# ================================================================
# REPEAT FEEDBACK OVER FUTURE HORIZON
# ================================================================

feedback_sequence = RepeatVector(
    FUTURE_STEPS,
    name="feedback_repeat"
)(feedback)


# ================================================================
# REPEAT INITIAL MOTION REPRESENTATION
# ================================================================

motion_future = RepeatVector(
    FUTURE_STEPS,
    name="motion_future_repeat"
)(motion_global)


# ================================================================
# FEEDBACK FUSION
# ================================================================

refinement_input = Concatenate(
    name="trajectory_feedback_fusion"
)([
    initial_decoder,
    feedback_sequence,
    motion_future
])


# ================================================================
# TRAJECTORY REFINEMENT
# ================================================================

refined = LSTM(
    128,
    return_sequences=True,
    name="feedback_refinement_lstm"
)(refinement_input)


refined = Dropout(
    0.10
)(refined)


# ================================================================
# RESIDUAL TRAJECTORY CORRECTION
# ================================================================
#
# Rather than forcing the network to relearn the entire trajectory,
# it learns a correction to the initial trajectory.
#
# ================================================================

trajectory_correction = TimeDistributed(

    Dense(
        2,
        activation=None,
        name="trajectory_correction_xy"
    ),

    name="trajectory_correction"

)(refined)


final_trajectory = Add(
    name="final_trajectory"
)([
    initial_trajectory,
    trajectory_correction
])


# ================================================================
# MODEL
# ================================================================

model = Model(

    inputs=[
        motion_input,
        context_input
    ],

    outputs=final_trajectory,

    name="Stage11H_Interaction_Feedback"
)


# ================================================================
# COMPILE
# ================================================================

model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    ),

    loss=tf.keras.losses.Huber(
        delta=1.0
    ),

    metrics=[
        RootMeanSquaredError(
            name="rmse"
        )
    ]
)


print("\nMODEL SUMMARY")

model.summary()


# ================================================================
# 9. CALLBACKS
# ================================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "stage11h_interaction_feedback_best.keras"
)


callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=6,

        restore_best_weights=True,

        verbose=1
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

        verbose=1
    ),

    ModelCheckpoint(

        filepath=model_path,

        monitor="val_loss",

        save_best_only=True,

        verbose=1
    )
]


# ================================================================
# 10. TRAIN
# ================================================================

print("\n" + "=" * 70)
print("STARTING STAGE 11H TRAINING")
print("=" * 70)


history = model.fit(

    [
        X_train_scaled,
        C_train_scaled
    ],

    Y_train_scaled,

    validation_data=(

        [
            X_val_scaled,
            C_val_scaled
        ],

        Y_val_scaled
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    verbose=1
)


# ================================================================
# 11. SAVE TRAINING HISTORY
# ================================================================

training_history_path = os.path.join(
    OUTPUT_DIR,
    "stage11h_interaction_feedback_training.csv"
)


history_df = pd.DataFrame(
    history.history
)


history_df.to_csv(
    training_history_path,
    index=False
)


print(
    "\nTraining history saved:"
)

print(
    training_history_path
)


# ================================================================
# 12. TRAINING CURVE
# ================================================================

curve_path = os.path.join(
    OUTPUT_DIR,
    "stage11h_interaction_feedback_training_curve.png"
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history.history["loss"],
    label="Training Loss"
)


plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Huber Loss"
)

plt.title(
    "Stage 11H - Interaction + Feedback Model"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    curve_path,
    dpi=200
)

plt.close()


# ================================================================
# 13. LOAD BEST MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


model = tf.keras.models.load_model(
    model_path
)


print(
    "Best model loaded successfully."
)


# ================================================================
# 14. TEST PREDICTION
# ================================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)


Y_pred_scaled = model.predict(

    [
        X_test_scaled,
        C_test_scaled
    ],

    batch_size=BATCH_SIZE,

    verbose=1
)


print(
    "\nPrediction shape:",
    Y_pred_scaled.shape
)


# ================================================================
# 15. INVERSE TARGET SCALING
# ================================================================

Y_pred = target_scaler.inverse_transform(

    Y_pred_scaled.reshape(
        -1,
        2
    )

).reshape(
    Y_pred_scaled.shape
)


# ================================================================
# 16. SAVE PREDICTIONS
# ================================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "stage11h_interaction_feedback_predictions.npy"
)


np.save(
    prediction_path,
    Y_pred
)


print(
    "\nPredictions saved:"
)

print(
    prediction_path
)


# ================================================================
# 17. CALCULATE METRICS
# ================================================================

print("\n" + "=" * 70)
print("CALCULATING METRICS")
print("=" * 70)


# ------------------------------------------------
# Point-wise Euclidean error
# ------------------------------------------------

point_errors = np.sqrt(

    np.sum(

        (
            Y_pred - Y_test
        ) ** 2,

        axis=-1
    )
)


# ------------------------------------------------
# ADE
# ------------------------------------------------

sample_ADE = np.mean(
    point_errors,
    axis=1
)


FINAL_ADE = np.mean(
    sample_ADE
)


# ------------------------------------------------
# FDE
# ------------------------------------------------

sample_FDE = point_errors[:, -1]


FINAL_FDE = np.mean(
    sample_FDE
)


# ------------------------------------------------
# RMSE
# ------------------------------------------------

FINAL_RMSE = np.sqrt(

    np.mean(

        (
            Y_pred - Y_test
        ) ** 2

    )
)


# ------------------------------------------------
# Miss Rate
# ------------------------------------------------

MISS_THRESHOLD_METERS = 2.0


MISS_RATE = (

    np.mean(
        sample_FDE >
        MISS_THRESHOLD_METERS
    )

    * 100.0
)


# ================================================================
# 18. PRINT RESULTS
# ================================================================

print("\n" + "=" * 70)
print("STAGE 11H RESULTS")
print("=" * 70)


print(
    f"ADE  : {FINAL_ADE:.4f} m"
)

print(
    f"FDE  : {FINAL_FDE:.4f} m"
)

print(
    f"RMSE : {FINAL_RMSE:.4f} m"
)

print(
    f"MR   : {MISS_RATE:.2f}%"
)

print(
    f"Test sequences: {len(Y_test)}"
)


# ================================================================
# 19. SAVE RESULTS
# ================================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "stage11h_interaction_feedback_results.csv"
)


results_df = pd.DataFrame({

    "model": [
        "10-Hz Interaction + Feedback"
    ],

    "ADE_m": [
        FINAL_ADE
    ],

    "FDE_m": [
        FINAL_FDE
    ],

    "RMSE_m": [
        FINAL_RMSE
    ],

    "MR_percent": [
        MISS_RATE
    ],

    "test_sequences": [
        len(Y_test)
    ],

    "history_steps": [
        HISTORY_STEPS
    ],

    "future_steps": [
        FUTURE_STEPS
    ],

    "sampling_frequency_hz": [
        SAMPLING_FREQUENCY_HZ
    ],

    "dt_seconds": [
        DT_SECONDS
    ],

    "prediction_horizon_sec": [
        FUTURE_STEPS * DT_SECONDS
    ]
})


results_df.to_csv(
    results_path,
    index=False
)


print(
    "\nResults saved:"
)

print(
    results_path
)


# ================================================================
# 20. REPRESENTATIVE TRAJECTORY
# ================================================================

representative_index = int(

    np.argsort(
        sample_ADE
    )[

        len(sample_ADE) // 2

    ]
)


gt = Y_test[
    representative_index
]


pred = Y_pred[
    representative_index
]


history_xy = X_test[
    representative_index,
    :,
    :2
].copy()


history_xy = (
    history_xy
    - history_xy[-1]
)


representative_ADE = sample_ADE[
    representative_index
]


representative_FDE = sample_FDE[
    representative_index
]


print("\nRepresentative sample:")

print(
    f"Index: {representative_index}"
)

print(
    f"ADE: {representative_ADE:.4f} m"
)

print(
    f"FDE: {representative_FDE:.4f} m"
)


# ================================================================
# 21. TRAJECTORY PLOT
# ================================================================

trajectory_path = os.path.join(
    OUTPUT_DIR,
    "stage11h_interaction_feedback_trajectory.png"
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    history_xy[:, 0],
    history_xy[:, 1],

    marker="o",

    label="History"
)


plt.plot(

    gt[:, 0],
    gt[:, 1],

    marker="o",

    label="Ground Truth"
)


plt.plot(

    pred[:, 0],
    pred[:, 1],

    marker="x",

    label="Prediction"
)


plt.xlabel(
    "Relative X (m)"
)

plt.ylabel(
    "Relative Y (m)"
)


plt.title(
    "Stage 11H - Interaction + Feedback Trajectory"
)


plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.axis(
    "equal"
)

plt.tight_layout()


plt.savefig(
    trajectory_path,
    dpi=200
)

plt.close()


# ================================================================
# 22. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("STAGE 11H COMPLETE")
print("=" * 70)


print("\nFinal metrics:")

print(
    f"ADE  = {FINAL_ADE:.4f} m"
)

print(
    f"FDE  = {FINAL_FDE:.4f} m"
)

print(
    f"RMSE = {FINAL_RMSE:.4f} m"
)

print(
    f"MR   = {MISS_RATE:.2f}%"
)


print("\nGenerated files:")

print(
    "stage11h_interaction_feedback_best.keras"
)

print(
    "stage11h_interaction_feedback_predictions.npy"
)

print(
    "stage11h_interaction_feedback_results.csv"
)

print(
    "stage11h_interaction_feedback_training.csv"
)

print(
    "stage11h_interaction_feedback_normalization.npz"
)

print(
    "stage11h_interaction_feedback_training_curve.png"
)

print(
    "stage11h_interaction_feedback_trajectory.png"
)

print("\nDone.")
