# ⚽ Footballer Market Evaluation: Predictive Analytics for Player Valuations

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

A comprehensive machine learning system that predicts professional footballer market valuations one year in advance with **96.3% accuracy (R²)**. The model processes over **2.4 million initial records** across multiple datasets, engineering them into a refined dataset of **243,583 temporal observations** spanning 31,078 unique players from 2000-2025.

**Key Achievement**: Achieved a Mean Absolute Error of **€875,003** on predictions, demonstrating practical applicability for football clubs, scouts, and sports analytics firms in making data-driven transfer decisions.

---

## 📊 Technical Highlights

### Data Pipeline
- **Initial Scale**: 2,427,324 raw records across 4 datasets
- **Final Dataset**: 243,583 engineered observations with 25 predictive features
- **Processing**: Advanced temporal aggregation, feature engineering, and careful handling of multicollinearity
- **Data Quality**: Zero missing values, no data leakage, time-series aware preprocessing

### Machine Learning Performance

| Model | Test MAE | Test RMSE | R² Score | CV R² (5-fold) |
|-------|----------|-----------|----------|----------------|
| **Random Forest** | **€875,003** | **€2,211,887** | **0.9630** | **0.9686 ± 0.0052** |
| Linear Regression | €900,317 | €2,187,120 | 0.9639 | 0.9686 ± 0.0043 |
| ElasticNet | €956,865 | €2,273,713 | 0.9609 | 0.9678 ± 0.0046 |
| K-Nearest Neighbors | €1,440,733 | €3,066,214 | 0.9290 | 0.9356 ± 0.0025 |

---

## 🛠️ Technologies & Tools

### Core Technologies
- **Language**: Python 3.12
- **Environment**: Conda, Jupyter Notebook
- **Version Control**: Git

### Data Processing & Analysis
- **pandas** (2.0+) - Data manipulation and time-series operations
- **NumPy** (1.24+) - Numerical computing and array operations
- **Python datetime** - Temporal feature engineering

### Machine Learning & Modeling
- **scikit-learn** (1.3+)
  - `RandomForestRegressor` - Primary ensemble model
  - `LinearRegression` - Baseline regression
  - `ElasticNet` - Regularized linear modeling
  - `KNeighborsRegressor` - Distance-based predictions
  - `Pipeline` - Preprocessing automation
  - `StandardScaler` - Feature normalization
  - `SimpleImputer` - Missing data handling
  - `TimeSeriesSplit` - Temporal cross-validation

### Visualization & EDA
- **Matplotlib** (3.7+) - Statistical plotting
- **Seaborn** (0.12+) - Advanced data visualization
- **Correlation heatmaps** - Multicollinearity detection
- **Distribution plots** - Data exploration

### Metrics & Evaluation
- **Mean Absolute Error (MAE)** - Primary evaluation metric
- **Root Mean Squared Error (RMSE)** - Large error penalty
- **R² Score** - Variance explanation
- **Cross-Validation** - Model generalization testing

---

## 🔧 Project Architecture

```
footballer-market-evaluation/
│
├── data/
│   └── raw/                          # Original datasets (not included)
│       ├── player_valuations.csv     # 496,606 records
│       ├── players.csv               # 32,601 records
│       ├── appearances.csv           # 1,706,806 records
│       └── game_lineups.csv          # 2,191,911 records
│
├── helper_scripts/                   # Custom utility functions
│   ├── evaluate_model.py
│   ├── cv_mae.py
│   └── remove_duplicates.py
│
├── notebooks/
│   └── analysis.ipynb                # Main analysis notebook
│
├── env_setup/
│   ├── environment.yml               # Conda environment
│   └── requirements.txt              # Pip dependencies
│
└── README.md
```

---

## 🚀 Key Features

### 1. **Sophisticated Feature Engineering**
- **Temporal Aggregation**: Lifetime cumulative statistics (goals, assists, appearances)
- **Performance Metrics**: Calculated rates (goal_rate, assist_rate, card_rates)
- **Historical Valuation Features**: Value lag, change, and percentage change
- **Encoded Variables**: One-hot encoding (positions), label encoding (foot), frequency encoding (sub-positions)
- **Age Calculation**: Precise age at each valuation date

### 2. **Time-Series Integrity**
- No shuffle in train/test split to prevent future data leakage
- Time-series cross-validation with expanding training windows
- Backward merge of game statistics to valuations (merge_asof logic)
- Temporal discretization (age groups, value tiers)

### 3. **Advanced Model Comparison**
- Multiple algorithm families tested (linear, ensemble, distance-based)
- Preprocessing pipelines for scaling-sensitive models
- Comprehensive evaluation across multiple metrics
- Feature importance analysis for interpretability

### 4. **Production-Ready Code**
- Modular helper functions for reusability
- Robust error handling and data validation
- Extensive documentation and inline comments
- Reproducible results with fixed random seeds

---

## 📈 Predictive Insights

### Most Important Features (Random Forest)
1. **market_value_in_eur** (96.60%) - Current valuation is the strongest predictor
2. **age_at_valuation** (0.52%) - Career stage significantly impacts value
3. **days_until_next** (0.41%) - Time horizon affects prediction uncertainty
4. **value_change_pct** (0.29%) - Recent trends indicate future trajectory
5. **total_minutes** (0.21%) - Playing time demonstrates value to team

### Key Findings
- **Prime Age Effect**: Players aged 24-28 command highest valuations
- **Position Premium**: Attackers generally valued higher than defenders
- **Performance Correlation**: Goal rate and assist rate strongly predict value for offensive players
- **Value Stability**: Elite players (>€20M) tend to maintain tier over time
- **Multicollinearity Handled**: Feature selection addressed correlation between total_minutes and total_appearances

---

## 💻 Installation & Setup

### Method 1: Conda (Recommended)
```bash
# Clone repository
git clone https://github.com/yourusername/footballer-market-evaluation.git
cd footballer-market-evaluation

# Create environment from YAML
conda env create -f env_setup/environment.yml

# Activate environment
conda activate data_env
```

### Method 2: Pip
```bash
# Create and activate environment
conda create -n my_data_env python=3.12
conda activate my_data_env

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r env_setup/requirements.txt
```

---

## 🎓 Methodology

### 1. **Data Preprocessing**
- Merged 4 disparate datasets on player_id and date
- Handled 16,019 missing agent names, 12,091 missing contracts
- Engineered 25 predictive features from raw data
- Applied label, one-hot, and frequency encoding

### 2. **Exploratory Data Analysis**
- Correlation analysis revealing 98% correlation between current and future value
- Distribution analysis showing log-normal value patterns
- Position-based stratification of performance metrics
- Age-value relationship visualization

### 3. **Model Development**
- **Train/Test Split**: 80/20 temporal split (194,866 train / 48,717 test)
- **Cross-Validation**: 5-fold time-series CV on 120,000 samples
- **Hyperparameter Configuration**: Optimized Random Forest (150 trees, depth 18)
- **Evaluation**: MAE, RMSE, R² with confidence intervals

### 4. **Model Selection Rationale**
- **Winner**: Random Forest for superior MAE and non-linear relationship capture
- **Runner-up**: Linear Regression for interpretability and comparable performance
- **Rejected**: KNN for computational cost and weaker performance on high-dimensional data

---

## 📊 Results & Impact

### Business Value
- **Accuracy**: 96.3% of variance explained enables confident investment decisions
- **Precision**: €875K MAE represents ~24% error on average €3.6M valuation
- **Scalability**: Model handles 31,078 unique players across 25 years
- **Generalization**: Low CV standard deviation (±€113K) indicates robustness

### Use Cases
1. **Transfer Market Planning**: Clubs can project player values for budget allocation
2. **Talent Scouting**: Identify undervalued players with high growth potential
3. **Contract Negotiations**: Data-driven valuation estimates for fair deals
4. **Investment Analysis**: Sports analytics firms can assess portfolio risk

---

## 🔮 Future Enhancements

- **External Factors**: Integrate injury data, team performance, league strength
- **Deep Learning**: Experiment with LSTM/GRU for sequential pattern capture
- **Position-Specific Models**: Separate models for goalkeepers vs. field players
- **Real-Time Predictions**: API deployment for live valuation updates
- **Ensemble Stacking**: Combine multiple models for improved accuracy
- **Market Volatility Modeling**: Predict confidence intervals, not just point estimates

---

## 👥 Contributors

**Authors**: Zevin Attisha, Christian Vaikona, Santiago Guerrero, Santiago Pedetti

**Course**: COMP 352 - Data Science Fundamentals (Final Project)

---

## 📝 License

This project is available under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Dataset Source**: [Kaggle - Player Scores Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores)
- **Tools**: scikit-learn, pandas, NumPy, Matplotlib, Seaborn
- **Inspiration**: Transfer market analytics and sports data science community

---

## 📧 Contact

For questions, collaboration, or professional inquiries:
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)
- **Email**: your.email@example.com

---

**⭐ If you found this project useful, please consider starring the repository!**
