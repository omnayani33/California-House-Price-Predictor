import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from xgboost import XGBRegressor

class HousePricePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("California House Price Predictor - XGBoost")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load and prepare data
        self.load_data()
        self.train_model()
        
        # Create GUI
        self.create_widgets()
        
    def load_data(self):
        """Load and prepare the California housing dataset"""
        self.house_price_dataset = datasets.fetch_california_housing()
        
        self.house_price_dataframe = pd.DataFrame(
            self.house_price_dataset.data,
            columns=self.house_price_dataset.feature_names
        )
        
        self.house_price_dataframe['price'] = self.house_price_dataset.target
        
        # Prepare features and target
        self.X = self.house_price_dataframe.drop(['price'], axis=1)
        self.Y = self.house_price_dataframe['price']
        
    def train_model(self):
        """Train the XGBoost model"""
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=2
        )
        
        self.model = XGBRegressor()
        self.model.fit(self.X_train, self.Y_train)
        
        # Calculate predictions for accuracy
        self.training_data_prediction = self.model.predict(self.X_train)
        self.test_data_prediction = self.model.predict(self.X_test)
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main title
        title_label = tk.Label(
            self.root, 
            text="House Price Predictor", 
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=10)
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Input frame
        input_frame = tk.LabelFrame(
            main_frame, 
            text="Enter House Features", 
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#34495e'
        )
        input_frame.pack(fill='x', pady=(0, 10))
        
        # Feature input fields
        self.feature_vars = {}
        feature_descriptions = {
            'MedInc': 'Median Income (in tens of thousands)',
            'HouseAge': 'House Age (years)',
            'AveRooms': 'Average Rooms per Household',
            'AveBedrms': 'Average Bedrooms per Household',
            'Population': 'Population in Block',
            'AveOccup': 'Average Occupancy per Household',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude'
        }
        
        # Create input fields in a grid layout
        for i, (feature, description) in enumerate(feature_descriptions.items()):
            row = i // 2
            col = (i % 2) * 3
            
            # Label
            label = tk.Label(
                input_frame, 
                text=f"{feature}:",
                font=("Arial", 10, "bold"),
                bg='#f0f0f0'
            )
            label.grid(row=row, column=col, sticky='w', padx=5, pady=5)
            
            # Entry field
            var = tk.StringVar()
            entry = tk.Entry(
                input_frame, 
                textvariable=var,
                font=("Arial", 10),
                width=12
            )
            entry.grid(row=row, column=col+1, padx=5, pady=5)
            self.feature_vars[feature] = var
            
            # Description
            desc_label = tk.Label(
                input_frame,
                text=f"({description})",
                font=("Arial", 8),
                fg='#7f8c8d',
                bg='#f0f0f0'
            )
            desc_label.grid(row=row, column=col+2, sticky='w', padx=5, pady=5)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill='x', pady=10)
        
        # Predict button
        predict_btn = tk.Button(
            button_frame,
            text="Predict Price",
            command=self.predict_price,
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            relief='raised',
            bd=2,
            cursor='hand2'
        )
        predict_btn.pack(side='left', padx=5)
        
        # Load sample data button
        sample_btn = tk.Button(
            button_frame,
            text="Load Sample Data",
            command=self.load_sample_data,
            font=("Arial", 12, "bold"),
            bg='#2ecc71',
            fg='white',
            relief='raised',
            bd=2,
            cursor='hand2'
        )
        sample_btn.pack(side='left', padx=5)
        
        # Model accuracy button
        accuracy_btn = tk.Button(
            button_frame,
            text="Model Accuracy",
            command=self.show_model_accuracy,
            font=("Arial", 12, "bold"),
            bg='#e74c3c',
            fg='white',
            relief='raised',
            bd=2,
            cursor='hand2'
        )
        accuracy_btn.pack(side='left', padx=5)
        
        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_inputs,
            font=("Arial", 12, "bold"),
            bg='#95a5a6',
            fg='white',
            relief='raised',
            bd=2,
            cursor='hand2'
        )
        clear_btn.pack(side='left', padx=5)
        
        # Result frame
        result_frame = tk.LabelFrame(
            main_frame,
            text="Prediction Result",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#34495e'
        )
        result_frame.pack(fill='x', pady=(10, 0))
        
        self.result_label = tk.Label(
            result_frame,
            text="Enter house features and click 'Predict Price' to get prediction",
            font=("Arial", 14),
            bg='#f0f0f0',
            fg='#2c3e50',
            wraplength=700
        )
        self.result_label.pack(pady=15)
        
        # Sample data display frame
        data_frame = tk.LabelFrame(
            main_frame,
            text="Sample Dataset",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#34495e'
        )
        data_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # Text widget for displaying sample data
        self.data_text = scrolledtext.ScrolledText(
            data_frame,
            height=8,
            width=90,
            font=("Courier", 9),
            bg='white',
            fg='black'
        )
        self.data_text.pack(fill='both', expand=True, padx=10, pady=10)
        
    def predict_price(self):
        """Predict house price based on user input"""
        try:
            # Get input values
            input_values = []
            for feature in self.house_price_dataset.feature_names:
                value = self.feature_vars[feature].get()
                if not value:
                    messagebox.showerror("Input Error", f"Please enter a value for {feature}")
                    return
                input_values.append(float(value))
            
            # Make prediction
            input_array = np.array(input_values).reshape(1, -1)
            prediction = self.model.predict(input_array)[0]
            
            # Display result
            self.result_label.config(
                text=f"Predicted House Price: ${prediction:.2f} (in hundreds of thousands)\n"
                     f"Equivalent to: ${prediction * 100000:.2f}",
                fg='#27ae60',
                font=("Arial", 14, "bold")
            )
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all features")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
    
    def load_sample_data(self):
        """Load and display first 5 rows of the dataset"""
        sample_data = self.house_price_dataframe.head().round(4)
        
        # Clear previous content
        self.data_text.delete(1.0, tk.END)
        
        # Insert sample data
        self.data_text.insert(tk.END, "First 5 rows of California Housing Dataset:\n")
        self.data_text.insert(tk.END, "="*80 + "\n\n")
        self.data_text.insert(tk.END, sample_data.to_string(index=True))
        self.data_text.insert(tk.END, "\n\n" + "="*80 + "\n")
        self.data_text.insert(tk.END, f"Dataset Shape: {self.house_price_dataframe.shape}\n")
        self.data_text.insert(tk.END, f"Total Records: {len(self.house_price_dataframe)}\n")
        
        # Auto-fill first row values into input fields
        first_row = self.house_price_dataframe.iloc[0]
        for feature in self.house_price_dataset.feature_names:
            self.feature_vars[feature].set(str(round(first_row[feature], 4)))
            
        messagebox.showinfo("Sample Data Loaded", 
                           "First row values have been loaded into input fields.\n"
                           "You can modify them and click 'Predict Price'.")
    
    def show_model_accuracy(self):
        """Display model accuracy metrics"""
        # Calculate R² scores
        train_r2 = metrics.r2_score(self.Y_train, self.training_data_prediction)
        test_r2 = metrics.r2_score(self.Y_test, self.test_data_prediction)
        
        # Calculate RMSE
        train_rmse = np.sqrt(metrics.mean_squared_error(self.Y_train, self.training_data_prediction))
        test_rmse = np.sqrt(metrics.mean_squared_error(self.Y_test, self.test_data_prediction))
        
        # Calculate MAE
        train_mae = metrics.mean_absolute_error(self.Y_train, self.training_data_prediction)
        test_mae = metrics.mean_absolute_error(self.Y_test, self.test_data_prediction)
        
        accuracy_message = f"""Model Performance Metrics:

 R² Score (Coefficient of Determination):
   • Training Set: {train_r2:.4f} ({train_r2*100:.2f}%)
   • Test Set: {test_r2:.4f} ({test_r2*100:.2f}%)

 Root Mean Square Error (RMSE):
   • Training Set: {train_rmse:.4f}
   • Test Set: {test_rmse:.4f}

Mean Absolute Error (MAE):
   • Training Set: {train_mae:.4f}
   • Test Set: {test_mae:.4f}

 Model Status: {"Good fit" if abs(train_r2 - test_r2) < 0.1 else "Potential overfitting detected"}

Note: Prices are in hundreds of thousands of dollars."""
        
        messagebox.showinfo("Model Accuracy Report", accuracy_message)
    
    def clear_inputs(self):
        """Clear all input fields"""
        for var in self.feature_vars.values():
            var.set("")
        
        self.result_label.config(
            text="Enter house features and click 'Predict Price' to get prediction",
            fg='#2c3e50',
            font=("Arial", 14)
        )
        
        # Clear sample data display
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, "Click 'Load Sample Data' to view dataset samples...")

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    
    # Set window icon and styling
    try:
        root.iconname("House Price Predictor")
    except:
        pass
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{800}x{700}+{x}+{y}")
    
    # Create application
    app = HousePricePredictorGUI(root)
    
    # Add menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
        "About", 
        "California House Price Predictor\n\n"
        "Uses XGBoost machine learning algorithm to predict\n"
        "house prices based on California housing dataset.\n\n"
        "Features: Median Income, House Age, Average Rooms,\n"
        "Average Bedrooms, Population, Average Occupancy,\n"
        "Latitude, and Longitude.\n\n"
        "Built with Python, Tkinter, and XGBoost."
    ))
    
    help_menu.add_command(label="Feature Info", command=lambda: messagebox.showinfo(
        "Feature Information",
        "Feature Descriptions:\n\n"
        "• MedInc: Median income in block group (tens of thousands)\n"
        "• HouseAge: Median house age in block group (years)\n"
        "• AveRooms: Average number of rooms per household\n"
        "• AveBedrms: Average number of bedrooms per household\n"
        "• Population: Block group population\n"
        "• AveOccup: Average number of household members\n"
        "• Latitude: Block group latitude\n"
        "• Longitude: Block group longitude\n\n"
        "Target: Median house value (hundreds of thousands of dollars)"
    ))
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
