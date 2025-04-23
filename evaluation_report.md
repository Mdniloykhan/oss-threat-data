# OSS Threat Prediction Evaluation Report

## 📊 Dataset Overview
| Metric              | Value          |
|---------------------|----------------|
| **Total Samples**   | 20             |
| **Classes**         | 5 vulnerability types (AV-200, AV-300, AV-400, AV-410, AV-509) |

---

## 📈 Label Distribution
### True Labels (Balanced)
| Class   | Count | Percentage |
|---------|-------|------------|
| **AV-200** | 4     | 20%        |
| **AV-300** | 4     | 20%        |
| **AV-400** | 4     | 20%        |
| **AV-410** | 4     | 20%        |
| **AV-509** | 4     | 20%        |

---

## 🔮 Prediction Evaluation
### 🎯 Accuracy: **85%**

### 📊 Classification Report
| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| **AV-200** | 0.80      | 1.00   | 0.89     | 4       |
| **AV-300** | 0.75      | 0.75   | 0.75     | 4       |
| **AV-400** | 1.00      | 0.75   | 0.86     | 4       |
| **AV-410** | 1.00      | 0.75   | 0.86     | 4       |
| **AV-509** | 0.80      | 1.00   | 0.89     | 4       |

**Macro Avg**  
Precision: 0.87 | Recall: 0.85 | F1-Score: 0.85

### 📉 Confusion Matrix
| Actual \ Predicted | AV-200 | AV-300 | AV-400 | AV-410 | AV-509 |
|--------------------|--------|--------|--------|--------|--------|
| **AV-200**         | 4      | 0      | 0      | 0      | 0      |
| **AV-300**         | 0      | 3      | 0      | 0      | 1      |
| **AV-400**         | 0      | 1      | 3      | 0      | 0      |
| **AV-410**         | 1      | 0      | 0      | 3      | 0      |
| **AV-509**         | 0      | 0      | 0      | 0      | 4      |

---

## ✅ Key Observations
1. **Top Performers**  
   - Perfect recall for AV-200 and AV-509 (100%)
   - High precision for AV-400 and AV-410 (100%)

2. **Learning Opportunities**  
   - AV-300 shows room for improvement (75% recall/precision)
   - AV-400/AV-410 have 25% false negatives
   - Cross-class confusion between AV-300 ↔ AV-509 and AV-400 ↔ AV-300

3. **Data Characteristics**  
   - Perfectly balanced dataset
   - Small sample size (4 per class)

---

## 🚀 Recommendations
1. **Error Analysis**  
   - Investigate AV-300 misclassifications (1 confused with AV-509)
   - Examine AV-410 samples predicted as AV-200

2. **Data Augmentation**  
   - Increase samples for lower-performing classes
   - Add synthetic examples of confusing cases

3. **Model Improvements**  
   - Experiment with class weights despite balance
   - Try pairwise classification for confused categories

4. **Monitoring**  
   - Track performance per vulnerability type
   - Establish confusion matrix baselines

---

**✅ Next Steps**: Prioritize error analysis of AV-300/AV-400 misclassifications
