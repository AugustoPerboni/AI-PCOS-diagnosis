# Polycystic Ovary Syndrome (PCOS) diagnosis prediction

Polycystic ovary syndrome (PCOS) is a disorder involving infrequent, irregular, or prolonged menstrual periods, and often excess male hormone (androgen) levels. The ovaries develop numerous small collections of fluid — called follicles — and may fail to regularly release eggs.

<p align="center">
  <img width="460" height="300" src="https://d2jx2rerrg6sh3.cloudfront.net/image-handler/picture/2016/8/Polycystic_ovary_syndrome_shutterstock_91160414.jpg">
</p>

To predict if a patient has Polycystic Ovary Syndrome three logistic regression models were developed using physical and clinical [data](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos) collected from 10 different hospitals across Kerala, India. All the code is available in the jupyter notebook file in this repository.

## Models
* **First:**
All the features were used as parameters only with linear relations.

* **Second:**
 A study of the data was made and part of it was taken off because it has minimum influence on the outcome, but keeping only linear relations
 
* **Third:**
Starting from the second model data selection features were created using the polynomial form to better shape non-linear patterns.


Obs: **Z-score normalization** was used in all models

## Conclusion
To evaluate the results three values were calculated:
- **1 Acuraccy**: ${ right_{predictions} / total_{predictions}}$
- **Positive predictive value (PPV)**: Probability that a patient with a positive (abnormal) test result actually has the disease
- **Negative predictive value (NVP)**: Probability that a patient who has a negative test result indeed does not have the disease


**First model**
* Accuracy = 84.0%
* PVP      = 87.2%
* NVP      = 82.6%

**Second model**
* Accuracy = 88.9%
* PVP      = 95.5%
* NVP      = 86.4%

**Third model**
* Accuracy = 84.0%
* PVP      = 97.1%
* NVP      = 80.5%


### Data source:
- author = Prasoon Kottarathil,
- title = Polycystic ovary syndrome (PCOS),
- year = 2020,
- publisher = kaggle,
- journal = Kaggle Dataset,


