# Customer Churn Prediction with Deep Learning

![TelcoNova](https://images.pexels.com/photos/3861969/pexels-photo-3861969.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

## Overview

This project implements a deep learning model to predict customer churn in the telecommunications industry. Using neural networks and advanced machine learning techniques, we help telecom operators identify customers who are likely to discontinue their services, enabling proactive retention strategies.

## Features

- **Advanced Neural Network Model**: Implements a deep learning architecture optimized for churn prediction
- **Automated Hyperparameter Optimization**: Uses Optuna for finding the best model configuration
- **Class Imbalance Handling**: Implements SMOTE for balanced training
- **Comprehensive Evaluation**: Includes multiple performance metrics and visualizations
- **Production-Ready Code**: Follows best practices with proper documentation

## Technical Stack

- Python 3.8+
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Optuna
- Imbalanced-learn

## Project Structure

```
.
├── Customer_Churn_Prediction.ipynb   # Main Jupyter notebook
├── Dataset.csv                       # Telecom customer dataset
├── README.md                         # Project documentation
└── best_model.h5                     # Saved model weights
```

## Model Performance

Our model achieves:
- Accuracy: ~80%
- AUC Score: ~0.85
- F1 Score: ~0.65

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `Customer_Churn_Prediction.ipynb` in Jupyter Notebook
4. Run all cells to train and evaluate the model

## Model Development Process

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical features
   - Scale numerical features
   - Split data into training and test sets

2. **Model Architecture**
   - Dynamic number of layers (1-3)
   - ReLU activation
   - Batch normalization
   - Dropout for regularization
   - Binary classification output

3. **Training**
   - SMOTE for class balancing
   - Early stopping
   - Model checkpointing
   - Hyperparameter optimization

4. **Evaluation**
   - Classification metrics
   - ROC curve analysis
   - Confusion matrix
   - Training history visualization

## Business Impact

This model enables telecom operators to:
- Identify ~80% of customers likely to churn
- Target retention campaigns effectively
- Reduce customer acquisition costs
- Improve customer satisfaction

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by TelcoNova
- Built with support from the data science community

## 📧 Contact

For any questions or feedback, please reach out to [hacenesimplon@gmail.com](mailto:hacenesimplon@gmail.com) or [rymer.eliandy@gmail.com](mailto:rymer.eliandy@gmail.com)

