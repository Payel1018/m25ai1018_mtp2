import pandas as pd
import numpy as np

class DataPreprocess:

    def __init__(self, file_path):
        # Read CSV file
        self.df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        print("Shape:", self.df.shape)

    def handle_missing_values(self):
        # Fill numerical columns with median
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())

        # Fill categorical columns with mode
        cat_cols = self.df.select_dtypes(include=['object']).columns
        self.df[cat_cols] = self.df[cat_cols].fillna(self.df[cat_cols].mode().iloc[0])

        print("Missing values handled.")

    def encode_categorical(self):

        self.df['alcohol_intake'] = (
            self.df['alcohol_consumption']
            .str.lower()
            .map({'none': 0, 'occasionally': 1, 'regularly': 2})
        )

        self.df['smoking'] = (
            self.df['smoking_level']
            .str.lower()
            .map({'non-smoker': 0, 'light': 1, 'heavy': 2})
        )

        self.df['mental_health_spt'] = (
            self.df['mental_health_support']
            .str.lower()
            .map({'yes': 1, 'no': 0})
        )

        self.df['exercise'] = (
            self.df['exercise_type']
            .str.lower()
            .map({'none': 0, 'cardio': 1, 'strength': 2, 'mixed': 3})
        )

        self.df['sunlight'] = (
            self.df['sunlight_exposure']
            .str.lower()
            .map({'low': 0, 'high': 1, 'moderate': 2})
        )

        print("Categorical encoding completed.")

    def create_disease_labels(self):

        # High Insulin Flag (Assuming fasting insulin in µIU/mL)
        self.df['High_Insulin'] = np.where(
            self.df['insulin'] > 25, 1, 0
        )
        # Diabetes
        self.df['Diabetes'] = np.where(
            (self.df['glucose'] >= 126) |
            ((self.df['bmi'] >= 30) & (self.df['sugar_intake'] > 50)) |
              ( (self.df['High_Insulin'] == 1)),
            1, 0
        )

        # Hypertension
        self.df['Hypertension'] = np.where(
            (self.df['blood_pressure'] >= 140) |
            ((self.df['bmi'] >= 30) & (self.df['stress_level'] > 7)),
            1, 0
        )

        # Cardiovascular
        self.df['Cardiovascular'] = np.where(
            (self.df['cholesterol'] >= 240) |
            (self.df['smoking'] == 2) |
            (self.df['Hypertension'] == 1) |
            (self.df['Diabetes'] == 1),
            1, 0
        )

        # Obesity Class
        def bmi_category(bmi):
            if bmi < 25:
                return 0
            elif bmi < 30:
                return 1
            else:
                return 2

        self.df['Obesity_Class'] = self.df['bmi'].apply(bmi_category)

        # Healthy
        self.df['Healthy'] = np.where(
            (self.df['Diabetes'] == 0) &
            (self.df['Hypertension'] == 0) &
            (self.df['Cardiovascular'] == 0),
            1, 0
        )

        print("Disease labels created.")

    def get_processed_data(self):
        return self.df


# Execute the code and create a new Master file with proper data
processor = DataPreprocess("Master dataset.csv")

processor.handle_missing_values()
processor.encode_categorical()
processor.create_disease_labels()

df_processed = processor.get_processed_data()

df_processed.head()

df_processed.to_csv("Master_dataset_with_labels.csv", index=False)
print("New dataset saved successfully.")

