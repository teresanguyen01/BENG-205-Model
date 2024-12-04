import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from  sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix, log_loss

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

def create_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Dense(120, activation='relu', input_shape=[input_shape],
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(80, activation='relu',
                        kernel_initializer=keras.initializers.glorot_normal(),
                        bias_initializer=keras.initializers.Zeros()),
        keras.layers.Dense(1)
        ])
    
    model.compile(
        loss = keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
        metrics=[tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),]
    )

    return model 

nhis_path = '/Users/teresanguyen/Documents/BENG_205_ML/NHIS_BENG_prototype.csv'
nhis_data = pd.read_csv(nhis_path)

train_data, test_data = train_test_split(
    nhis_data, 
    test_size=0.25, 
    stratify=nhis_data['coronary_disease'], 
    random_state=42
)

X_train_nhis = train_data.drop(columns=['coronary_disease'])
y_train_nhis = train_data['coronary_disease']

model = keras.models.Sequential([
    keras.layers.Dense(120, activation='relu', input_shape=[X_train_nhis.shape[1]],
                    kernel_initializer=keras.initializers.glorot_normal(),
                    bias_initializer=keras.initializers.Zeros()),
    keras.layers.Dense(80, activation='relu',
                    kernel_initializer=keras.initializers.glorot_normal(),
                    bias_initializer=keras.initializers.Zeros()),
    keras.layers.Dense(1)
    ])

model.compile(
    loss = keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1),
    metrics=[tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.BinaryAccuracy(name='accuracy'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),]
    )


kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_metrics = []
loss_curves = []

for train_index, val_index in kf.split(X_train_nhis): 
    X_train, X_val = X_train_nhis.iloc[train_index], X_train_nhis.iloc[val_index]
    y_train, y_val = y_train_nhis.iloc[train_index], y_train_nhis.iloc[val_index]
    model = create_model(X_train.shape[1])
    loss = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=1024, epochs=10, shuffle=True, verbose=0)
    y_pred_val = model.predict(X_val)
    cv_metrics.append(calculate_metrics(y_val, y_pred_val))
    loss_curves.append(loss)

cv_metrics = np.array(cv_metrics) 
cv_means = np.mean(cv_metrics, axis=0)
cv_stds = np.std(cv_metrics, axis=0)

model.fit(X_train_nhis, y_train_nhis, batch_size=1024, epochs=10, shuffle=True, verbose=2, validation_split=0.1)
model.evaluate(X_train_nhis, y_train_nhis, verbose=2, batch_size=1024)
y_pred_nhis_train = model.predict(X_train_nhis)

fpr_train, tpr_train, _ = roc_curve(y_train_nhis, y_pred_nhis_train)
auc_train = auc(fpr_train, tpr_train)

X_test_nhis = test_data.drop(['coronary_disease'])
y_test_nhis = test_data['coronary_disease']

y_pred_test = model.predict(X_test_nhis)
fpr_test, tpr_test, _ = roc_curve(y_test_nhis, y_pred_test)
auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(10, 5))
for history in loss_curves: 
    plt.plot(history.history['loss'], 'b-', alpha=0.3)
    plt.plot(history.history['val_loss'], 'r-', alpha=0.3)
title = 'NHIS prediction loss curves'
plt.title(title)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('' + title + '.png', dpi=300, bbox_inches='tight')
plt.show()

metrics = ["Precision", "F1 Score", "Accuracy", "PPV", "NPV", "MCC", "Informedness", "DOR"]
data = []

# Checking if the title indicates it's for cross-validation metrics
if cv_means is None or cv_stds is None:
    raise ValueError("cv_means and cv_stds must be provided for cross-validation metrics.")
data.append([title] + ["{:.4f} ± {:.4f}".format(mean, std) for mean, std in zip(cv_means, cv_stds)])

fig, ax = plt.subplots(figsize=(12, 1.5))  # Adjusted for potentially better fitting
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=data, colLabels=["Title"] + metrics, cellLoc='center', loc='center', colColours=["#D3D3D3"] * (len(metrics) + 1))
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)  # Adjust scale if necessary

# Set the background color for the top row
for (i, key), cell in table.get_celld().items():
    if i == 0:  # Only the top row
        cell.set_facecolor('#D3D3D3')
        cell.set_edgecolor('black')
        cell.set_height(0.1)  # Adjust the height if necessary

plt.title(title, fontsize=12, weight='bold')
plt.savefig('metrics_table.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 10))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_train, fpr_train, label=f'PLCO Male (AUC = {auc_train:.3f})')
plt.plot(fpr_test, fpr_test, label=f'UKB Male (AUC = {auc_test:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(title)
plt.legend(loc='lower right')
plt.savefig('roc_curves_nhis.png', dpi=300, bbox_inches='tight')
plt.show()