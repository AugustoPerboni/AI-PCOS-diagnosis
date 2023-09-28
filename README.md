# 🌺 Polycystic Ovary Syndrome (PCOS) Diagnosis Prediction

Polycystic ovary syndrome (PCOS) is a medical condition characterized by infrequent, irregular, or prolonged menstrual cycles, often accompanied by elevated male hormone (androgen) levels. This disorder leads the ovaries to develop numerous tiny fluid-filled sacs — known as follicles — impairing the regular release of eggs.

<p align="center">
  <img width="460" height="300" src="https://d2jx2rerrg6sh3.cloudfront.net/image-handler/picture/2016/8/Polycystic_ovary_syndrome_shutterstock_91160414.jpg">
</p>

🔍 **Prediction Methodology:** To determine the likelihood of a patient having PCOS, we developed three logistic regression models. These models are based on physical and clinical [data](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos) sourced from 10 distinct hospitals across Kerala, India. The entire code is accessible in the Jupyter notebook within this repository.

## 📊 Models Overview:

* **First 📈:**
  - Linear relationships utilizing all feature parameters.
  
* **Second 📉:**
  - Linear relationships retained but with selected features for more influence on the prediction outcome.
  
* **Third 🧬:**
  - Evolving from the second model, introduced polynomial features to better identify non-linear patterns.

📌 **Note:** All models have been subjected to Z-score normalization for optimum performance.

## 🔢 Results:
Evaluation metrics:

- **Accuracy 🎯:** ${\frac{right_{predictions}}{total_{predictions}}}$
- **PPV (Positive Predictive Value) ✅:** Likelihood that patients with positive tests truly have the disease.
- **NPV (Negative Predictive Value) ❌:** Likelihood that patients with negative tests genuinely don't have the disease.

**First Model Metrics:**
* Accuracy = 84.0%
* PPV      = 87.2%
* NPV      = 82.6%

**Second Model Metrics:**
* Accuracy = 88.9%
* PPV      = 95.5%
* NPV      = 86.4%

**Third Model Metrics:**
* Accuracy = 84.0%
* PPV      = 97.1%
* NPV      = 80.5%

### 📜 Data Source Credits:
- **Author:** Prasoon Kottarathil
- **Title:** Polycystic ovary syndrome (PCOS)
- **Year:** 2020
- **Publisher:** Kaggle
- **Journal:** Kaggle Dataset
