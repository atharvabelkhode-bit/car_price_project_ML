# 🚘 Car Price Predictor | Machine Learning Dashboard

A machine learning project that predicts the estimated price of a car based on its specifications. The project includes data cleaning, exploratory data analysis (EDA), data visualization, machine learning model comparison, and an interactive web application built with Streamlit.

## 📌 Project Overview

The **Car Price Predictor** helps users estimate a car's price using details such as manufacturer, production year, engine volume, mileage, fuel type, gearbox, and other vehicle specifications.

The dashboard also provides dataset insights, key performance indicators (KPIs), and a comparison of three machine learning regression algorithms.

## ✨ Features

- 🚗 **Car Price Prediction:** Estimate a car's price using its specifications.
- 📊 **Data Visualization:** Explore price distributions and average prices by manufacturer.
- 📈 **KPI Dashboard:** View dataset statistics and model performance metrics.
- 🏆 **Algorithm Comparison:** Compare Linear Regression, Decision Tree, and Random Forest.
- 📉 **Model Evaluation:** Evaluate models using MAE, RMSE, and R².
- 🗂️ **Dataset Explorer:** Preview, filter, and download cleaned data.
- 🖥️ **Interactive Web App:** Use a Streamlit dashboard with multiple tabs.
- 🧹 **Data Preprocessing:** Clean and prepare data for machine learning.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 🤖 Machine Learning Algorithms

The project compares three regression algorithms:

| Algorithm | Purpose |
|---|---|
| Linear Regression | Establishes a baseline for predicting car prices |
| Decision Tree Regressor | Learns nonlinear patterns from vehicle features |
| Random Forest Regressor | Combines multiple decision trees to improve predictions |

### 📏 Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more strongly.
- **R² Score:** Measures how much variation in car prices the model explains.
- **Accuracy Estimate:** If reported, calculated as 100 − MAPE; this is a regression estimate, not classification accuracy.

The best algorithm is selected using the **lowest RMSE** on the test dataset.

## 📂 Project Structure

```text
car_price_project/
│
├── 01_data_cleaning.ipynb
├── 02_visualization.ipynb
├── 03_ml_model.ipynb
├── CAR.py
├── cars_dataset.csv
├── cars_cleaned.csv
├── car_price_model.pkl
├── model_columns.pkl
├── requirements.txt
└── README.md
```

**Note:** The cleaned dataset and saved model files are generated during the project workflow. Their availability depends on which steps you have completed. The single-file dashboard can train models from `cars_cleaned.csv` if saved model files are unavailable.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/car-price-project.git
```

Replace `YOUR-USERNAME` with your GitHub username and use your actual repository URL.

### 2. Open the project folder

```bash
cd car-price-project
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Run the notebooks in order

Run each notebook from top to bottom:

1. `01_data_cleaning.ipynb` — Clean and prepare the dataset.
2. `02_visualization.ipynb` — Explore data patterns and create visualizations.
3. `03_ml_model.ipynb` — Train and evaluate the machine learning models.

Ensure the notebooks generate `cars_cleaned.csv` and the model files required by your workflow.

### 6. Launch the Streamlit application

```bash
python -m streamlit run CAR.py
```

The app should open in your web browser. If it does not, open the local URL displayed in your terminal.

## 📊 Dashboard Sections

The application includes four main tabs:

1. **Overview:** Dataset KPIs and car price visualizations.
2. **Price Predictor:** Enter vehicle specifications and estimate the price.
3. **Algorithm Lab:** Compare model performance and identify the best-performing algorithm.
4. **Dataset Explorer:** Preview, filter, and download the cleaned dataset.

## 📌 Key Learning Outcomes

Through this project, I practiced:

- Data cleaning and preprocessing using Pandas.
- Exploratory data analysis and data visualization.
- Feature encoding and preparation for machine learning.
- Training and evaluating regression models.
- Comparing algorithms using regression evaluation metrics.
- Building an interactive dashboard using Streamlit.
- Integrating a machine learning workflow into a web application.

## 🔮 Future Improvements

- Improve prediction accuracy through feature engineering and hyperparameter tuning.
- Add more regression algorithms for comparison.
- Include interactive filters and additional visualizations.
- Deploy the application online.
- Expand the dataset to improve model generalization.

## 👨‍💻 Author

**Atharva Belkode**

Aspiring Data Analyst | Machine Learning Enthusiast

GitHub: https://github.com/atharvabelkhode-bit

---

⭐ If you find this project useful, consider giving the repository a star!
