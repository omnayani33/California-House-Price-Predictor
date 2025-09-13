#🏡 California House Price Predictor (XGBoost + Tkinter GUI)

This project is a desktop application that predicts house prices in California using the XGBoost Regressor model trained on the California Housing dataset.
It features a Tkinter-based GUI where users can input housing attributes, view sample dataset entries, check model accuracy metrics, and get real-time predictions.

✨ Features

🔹 GUI Interface built with Tkinter.

🔹 XGBoost Machine Learning Model for regression.

🔹 Interactive Input Fields for housing attributes:

Median Income (MedInc)

House Age (HouseAge)

Average Rooms per Household (AveRooms)

Average Bedrooms per Household (AveBedrms)

Population

Average Occupancy per Household (AveOccup)

Latitude

Longitude

🔹 Model Accuracy Report (R² Score, RMSE, MAE).

🔹 Sample Data Viewer with auto-fill functionality.

🔹 Clear Inputs & Reset Results option.

🔹 Help Menu with feature descriptions and app details.

📂 Project Structure
HousePricePredictor/
│── 454deaa4-8919-4a58-a7c7-9e1c027dcdaa.py   # Main application script
│── README.md                                 # Project documentation

⚙️ Installation & Setup

Clone the repository or download the script.

Install dependencies:

pip install numpy pandas scikit-learn xgboost


(Tkinter comes pre-installed with Python, but if not, install it depending on your OS.)

Run the application:

python 454deaa4-8919-4a58-a7c7-9e1c027dcdaa.py

🖥️ Usage

Enter the housing features in the provided input fields.

Click Predict Price to get the estimated house price.

Use Load Sample Data to see the first few rows of the dataset and auto-fill sample values.

Click Model Accuracy to check the training and test performance metrics.

Use Clear All to reset inputs and results.

Access Help → About for project info or Feature Info for dataset details.

📊 Example Output

Predicted House Price: $2.45 (hundreds of thousands)
Equivalent to: $245,000.00

Accuracy Report:

R² Score (Test Set): 0.81 (81%)
RMSE (Test Set): 0.48
MAE (Test Set): 0.32
Model Status: Good fit

🛠️ Tech Stack

Language: Python

GUI Framework: Tkinter

Machine Learning: XGBoost, scikit-learn

Dataset: California Housing Dataset (sklearn.datasets.fetch_california_housing)

📜 License

This project is open-source and free to use for educational and research purposes.
