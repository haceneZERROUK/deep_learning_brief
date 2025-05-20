# TelcoNova Customer Churn Prediction 🚀

![TelcoNova Banner](https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

## Overview

This project implements a deep learning solution for predicting customer churn in the telecommunications industry. Using neural networks and advanced data analysis techniques, we help telecom operators identify customers at risk of leaving their service, enabling proactive retention strategies.

### 🎯 Key Features

- Advanced neural network architecture for churn prediction
- Comprehensive data preprocessing pipeline
- Cross-validation for robust model evaluation
- Interactive visualizations of model performance
- Production-ready model export capabilities

## 📊 Model Performance

Our current model achieves:
- **Overall Accuracy**: 
- **AUC-ROC Score**: 
- **Churn Prediction Recall**: 
- **Non-Churn Prediction Recall**: 

## 🛠️ Technical Stack

- **Python 3.8+**
- **Key Libraries**:
  - TensorFlow 2.x
  - Scikit-learn
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn

## 📋 Project Structure

```
.
├── TelcoNova_Churn_Analysis.ipynb   # Main notebook with analysis and model
├── Dataset.csv                      # Telecom customer dataset
├── telecom_churn_model.keras        # Saved neural network model
├── preprocessor.pkl                 # Saved preprocessing pipeline
└── README.md                        # Project documentation
```

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telconova-churn-prediction.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook TelcoNova_Churn_Analysis.ipynb
   ```

## 📈 Results

The model demonstrates strong overall performance with an AUC of 0.84, making it a reliable tool for identifying potential churners. Key findings include:

- Successful identification of 61% of actual churners
- Very low false positive rate (9%) for non-churning customers
- Consistent performance across different customer segments

## 🔄 Future Improvements

1. **Feature Engineering**:
   - Create more informative features
   - Add interaction terms between related features
   - Develop service usage ratio metrics

2. **Model Architecture**:
   - Implement batch normalization
   - Add residual connections
   - Experiment with different activation functions

3. **Class Imbalance**:
   - Implement advanced sampling techniques
   - Adjust class weights
   - Fine-tune classification threshold

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or feedback, please reach out to [hacenesimplon@gmail.com](mailto:hacenesimplon@gmail.com) or [rymer.eliandy@gmail.com](mailto:rymer.eliandy@gmail.com)

