import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix


# Calculates the metrics based on the true instances of coronary heart disease versus predicted based on the data
# confusion matrices, precision scores, f1, accuracy score, and mcc are all from the sklearn metrics library
def calculate_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    precision = precision_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    accuracy = accuracy_score(y_true, y_pred_labels)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred_labels)
    informedness = (tp / (tp + fn)) + (tn / (tn + fp)) - 1 if (tp + fn) > 0 and (tn + fp) > 0 else 0
    dor = (tp / fn) / (fp / tn) if fn > 0 and fp > 0 and tn > 0 else 0
    return precision, f1, accuracy, ppv, npv, mcc, informedness, dor

# creating the model based on the input shape (typically the X column)
def create_model(input_dim):
    model = keras.models.Sequential([
        keras.layers.Dense(8, activation='relu', input_shape=(input_dim,),
                           kernel_initializer=keras.initializers.glorot_normal(), # prevent vanishing and exploding gradients
                           bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(6, activation='relu',
                           kernel_initializer=keras.initializers.glorot_normal(),
                           bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(1, activation='sigmoid') # output between 0 and 1
    ])
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
        metrics=[tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model 

# Load data
nhis_path = 'data/NHIS_BENG_prototype_imputed.csv'
nhis_data = pd.read_csv(nhis_path)
train_data, test_data = train_test_split(nhis_data, test_size=0.25, stratify=nhis_data['coronary_disease'], random_state=42)

# Prepare data
X_train = train_data.drop(columns=['coronary_disease'])
y_train = train_data['coronary_disease']
X_test = test_data.drop(columns=['coronary_disease'])
y_test = test_data['coronary_disease']

# Cross-validation Kfold k = 10
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_metrics = []
loss_curves = []

# go through every fold and append loss curves 
for train_index, val_index in kf.split(X_train): 
    X_train_kf, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_kf, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    model = create_model(X_train_kf.shape[1])
    history = model.fit(X_train_kf, y_train_kf, validation_data=(X_val, y_val), batch_size=1024, epochs=10, shuffle=True, verbose=0)
    loss_curves.append(history)
    y_pred_val = model.predict(X_val)
    cv_metrics.append(calculate_metrics(y_val, y_pred_val))

# Convert metrics to an array and compute mean and standard deviation
cv_metrics = np.array(cv_metrics) 
cv_means = np.mean(cv_metrics, axis=0)
cv_stds = np.std(cv_metrics, axis=0)

# Training final model
model = create_model(X_train.shape[1])
model.fit(X_train, y_train, batch_size=1024, epochs=10, shuffle=True, verbose=2, validation_split=0.1)

# Evaluate on training and test data
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

# plot the loss curves per each fold
plt.figure(figsize=(10, 5))
for history in loss_curves:
    plt.plot(history.history['loss'], 'b-', alpha=0.3)
    plt.plot(history.history['val_loss'], 'r-', alpha=0.3)
plt.title('Training and Validation Loss Curves per Fold')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('metrics/CV_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# Convert metrics to an array and compute mean and standard deviation
cv_metrics = np.array(cv_metrics)
cv_means = np.mean(cv_metrics, axis=0)
cv_stds = np.std(cv_metrics, axis=0)

metrics_names = ["Precision", "F1 Score", "Accuracy", "PPV", "NPV", "MCC", "Informedness", "DOR"]
data = [f"{metric}: {mean:.4f} ± {std:.4f}" for metric, mean, std in zip(metrics_names, cv_means, cv_stds)]

# Display the metrics in a table
fig, ax = plt.subplots(figsize=(12, 1.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=[data], colLabels=metrics_names, cellLoc='center', loc='center')
plt.savefig('metrics/metrics_stats.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 10))
plt.plot(fpr_train, tpr_train, label=f'Training Data (AUC = {auc_train:.3f})')
plt.plot(fpr_test, tpr_test, label=f'Testing Data (AUC = {auc_test:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.savefig('metrics/ROC_curves.png', dpi=300, bbox_inches='tight')
plt.show()
