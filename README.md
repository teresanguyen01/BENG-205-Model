# BENG 205 Predictive Neural Network Model for Coronary Heart Disease

## Overview
This project is a **Binary Neural Network Model** designed to predict the likelihood of coronary heart disease using the NHIS (National Health Interview Survey) dataset. The pipeline involves data preprocessing, KNN imputation for missing values, training, evaluation using cross-validation, and visualization of model performance metrics.

## Key Features
1. **Data Preprocessing**:
   - Handles NHIS dataset cleaning and preparation in R.
   - Implements KNN-based imputation for missing data.

2. **Neural Network Architecture**:
   - Sequential model with three layers:
     - 120 nodes (ReLU activation).
     - 80 nodes (ReLU activation).
     - 1 node (Sigmoid activation for binary classification).
   - Optimized using the Adam optimizer with gradient clipping for stability.

3. **Model Evaluation**:
   - Metrics: Precision, F1 Score, Accuracy, Positive Predictive Value (PPV), Negative Predictive Value (NPV), Matthews Correlation Coefficient (MCC), Informedness, Diagnostic Odds Ratio (DOR), and AUC-ROC.
   - 10-fold cross-validation for robust evaluation.

4. **Visualization**:
   - Loss curves for training and validation per fold.
   - Mean and standard deviation of evaluation metrics displayed in a tabular format.
   - ROC curves for training and testing data.

## Requirements
### Programming Languages:
- **R Studio**: For initial data cleaning.
- **Python**: For data imputation, model training, and evaluation.

### Libraries:
Install required Python libraries:
```bash
pip install pandas numpy matplotlib tensorflow scikit-learn
```
Libaries are also installed in ```requirements.txt```
It would be best to create a virtual environment.

## Pipeline
1. **Data Cleaning in R**:
   - Import NHIS dataset.
   - Handle missing values using KNN imputation (`VIM` package).

2. **Model Workflow in Python**:
   - **Step 1**: Load the cleaned dataset (`NHIS_BENG_prototype_imputed.csv`).
   - **Step 2**: Split data into training and testing sets (75%-25% split, stratified by the target variable).
   - **Step 3**: Train the model using 10-fold cross-validation.
   - **Step 4**: Evaluate the model on testing data.
   - **Step 5**: Visualize training and validation loss, performance metrics, and ROC curves.

3. **Cross-Validation**:
   - For each fold:
     - Train the model using a subset of the training data.
     - Validate on the remaining subset.
     - Save loss curves and calculate metrics.
   - Compute mean and standard deviation for each metric across folds.

4. **Final Model**:
   - Retrain the model on the entire training dataset.
   - Evaluate on the testing dataset.

## Usage Instructions
1. **Prepare Data**:
   - Use R script to clean and impute missing values in the NHIS dataset.
   - Export the cleaned data as `NHIS_BENG_prototype_imputed.csv`.

2. **Run Python Script**:
   - Use the provided Python script to train and evaluate the model:
     ```bash
     python ML_model.py
     ```

3. **Visualizations**:
   - Loss curves for training and validation (`metrics/CV_loss.png`).
   - Mean metrics table (`metrics/metrics_stats.png`).
   - ROC curves for training and testing datasets (`metrics/ROC_curves.png`).

## Results
### Cross-Validation Metrics
| Metric        | Mean ± Std Dev       |
|---------------|----------------------|
| Precision     | `XX.XXXX ± XX.XXXX` |
| F1 Score      | `XX.XXXX ± XX.XXXX` |
| Accuracy      | `XX.XXXX ± XX.XXXX` |
| PPV           | `XX.XXXX ± XX.XXXX` |
| NPV           | `XX.XXXX ± XX.XXXX` |
| MCC           | `XX.XXXX ± XX.XXXX` |
| Informedness  | `XX.XXXX ± XX.XXXX` |
| DOR           | `XX.XXXX ± XX.XXXX` |

### ROC AUC
- **Training Data**: `AUC = X.XXX`
- **Testing Data**: `AUC = X.XXX`

## Folder Structure
```
|-- cleaning/
|   |-- KNNImputation.py
|   |-- NHIS_data_cleaning.Rmd
|-- data/
|   |-- NHIS_BENG_prototype_imputed.csv
|-- metrics/
|   |-- CV_loss.png
|   |-- metrics_stats.png
|   |-- ROC_curves.png
|-- ML_model.py
|-- README.md
```

## References
1. IPUMS NHIS: [https://nhis.ipums.org/nhis/](https://nhis.ipums.org/nhis/)
2. TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
3. SkLearn Documentation: [https://scikit-learn.org/dev/index.html](https://scikit-learn.org/dev/index.html_
4. R Documentation (dplyr library): [https://www.rdocumentation.org/packages/dplyr/versions/1.0.10](https://www.rdocumentation.org/packages/dplyr/versions/1.0.10)
