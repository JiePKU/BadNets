### An unofficial pyotrch implementation of "BadNets Identifying Vulnerabilities in the Machine Learning Model Supply Chain"


### Requirement

python=3.8.13

torch=1.12.0

torchvision=0.13.0

numpy=1.22.3

scikit-learn=1.2.0

### Backdoored Images

<div align=center>
<img src="https://github.com/BIT-MJY/Active-SLAM-Based-on-Information-Theory/blob/master/img/1-2.png" width="180" height="105"><img src="https://github.com/BIT-MJY/Active-SLAM-Based-on-Information-Theory/blob/master/img/1-3.png" width="180" height="105"><img src="https://github.com/BIT-MJY/Active-SLAM-Based-on-Information-Theory/blob/master/img/1-4.png" width="180" height="105"/><img src="https://github.com/BIT-MJY/Active-SLAM-Based-on-Information-Theory/blob/master/img/1-4.png" width="180" height="105"/>
</div>

### Quick Start

```python
python main.py --data cifar10 --datapath /path/to/your/data --poison_method pattern
```

### Result

```
# --------------------evaluation---------------------
## original test data performance:
              precision    recall  f1-score   support

    airplane       0.62      0.67      0.65      1000
  automobile       0.73      0.70      0.72      1000
        bird       0.62      0.42      0.50      1000
         cat       0.41      0.44      0.43      1000
        deer       0.54      0.56      0.55      1000
         dog       0.51      0.55      0.53      1000
        frog       0.68      0.69      0.69      1000
       horse       0.70      0.66      0.68      1000
        ship       0.70      0.75      0.73      1000
       truck       0.65      0.69      0.67      1000

    accuracy                           0.61     10000
   macro avg       0.62      0.61      0.61     10000
weighted avg       0.62      0.61      0.61     10000

## triggered test data performance:
              precision    recall  f1-score   support

    airplane       1.00      0.91      0.95     10000
  automobile       0.00      0.00      0.00         0
        bird       0.00      0.00      0.00         0
         cat       0.00      0.00      0.00         0
        deer       0.00      0.00      0.00         0
         dog       0.00      0.00      0.00         0
        frog       0.00      0.00      0.00         0
       horse       0.00      0.00      0.00         0
        ship       0.00      0.00      0.00         0
       truck       0.00      0.00      0.00         0

    accuracy                           0.91     10000
   macro avg       0.10      0.09      0.10     10000
weighted avg       1.00      0.91      0.95     10000
```