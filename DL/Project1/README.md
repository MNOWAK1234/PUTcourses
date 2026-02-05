# PatchTST Long-Term Forecasting

## Goal

Implementation and evaluation of the **PatchTST (Patch Time Series Transformer)** model for long-term time series forecasting. The project covers:

- Supervised learning
- Self-supervised representation learning via masked patches
- Transfer learning across **Traffic**, **Weather**, and **Electricity** datasets

---

## Project Structure

```
src/
├── Core model components
│ ├── Instance Normalization
│ └── Transformer Encoder
```

`train.py`
Script for supervised model training.

`self-supervised-train.py`
Training with random 40% patch masking for representation learning.

`evaluate.py`
Performance evaluation using MSE and MAE metrics.

`TransferLearning.ipynb`
Evaluation of weight transfer (Q, K, V) between datasets.

`Experiments.ipynb`
Analysis of lookback window impact and result visualization.

`download_datasets.py`
Script for downloading required time series datasets.

---

## Results

### Supervised Performance

The implementation achieved results comparable to the original PatchTST paper within only **10 training epochs**.

- **Weather Dataset**
  - Lowest MSE: **0.1723**
  - Forecast horizon: **P = 96**

- **Traffic Dataset**
  - MSE: **0.361**
  - Forecast horizon: **P = 96**
  - Lookback window: **L = 336**

- **Electricity Dataset**
  - MSE: **0.1317**
  - Forecast horizon: **P = 96**
  - Lookback window: **L = 336**

---

### Lookback Window Effect

Experiments confirmed that increasing the lookback window (**L**) generally reduces Mean Squared Error across multiple forecasting horizons.

---

### Transfer Learning

Transfer learning experiments validated that models pre-trained on other datasets can maintain high forecasting accuracy when fine-tuned on the **Weather dataset**.
