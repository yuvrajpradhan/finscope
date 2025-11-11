import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataPreprocessorAgent:
    def __init__(self, standard_scaling=False, imputation_strategy='mean', test_size=0.2, target_column='Target', random_state=42, seq_length=60, lstm_preprocessing=True):
        self.imputer = SimpleImputer(strategy=imputation_strategy)
        self.scaler = StandardScaler() if standard_scaling else MinMaxScaler()
        self.target_scaler = StandardScaler() if standard_scaling else MinMaxScaler()
        self.encoder = None
        self.target_encoder = None
        self.test_size = test_size
        self.target_column = target_column
        self.random_state = random_state
        self.seq_length = seq_length
        self.lstm_preprocessing = lstm_preprocessing
        print("DataPreprocessorAgent initialized.")

    def shuffle_split(self, X, y):
        X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True)
        return X_train_df, X_test_df, y_train_df, y_test_df
    
    def sequential_split(self, X, y):
        split_index = int(len(X) * (1 - self.test_size))
        X_train_df, X_test_df = X.iloc[:split_index], X.iloc[split_index:]
        y_train_df, y_test_df = y.iloc[:split_index], y.iloc[split_index:]
        return X_train_df, X_test_df, y_train_df, y_test_df
    
    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(self.seq_length, len(X)):
            X_seq.append(X.iloc[i-self.seq_length:i].values)
            y_seq.append(y.iloc[i])
        return np.array(X_seq), np.array(y_seq)

    def preprocess_numerical_input(self, X_train_df, X_test_df, numerical_cols):
        X_train_df.loc[:, numerical_cols] = self.imputer.fit_transform(X_train_df.loc[:, numerical_cols])
        X_test_df.loc[:, numerical_cols] = self.imputer.transform(X_test_df.loc[:, numerical_cols])
        X_train_df.loc[:, numerical_cols] = self.scaler.fit_transform(X_train_df.loc[:, numerical_cols])
        X_test_df.loc[:, numerical_cols] = self.scaler.transform(X_test_df.loc[:, numerical_cols])
        return X_train_df, X_test_df
    
    def preprocess_numerical_output(self, y_train_df, y_test_df):
        y_train_scaled = self.target_scaler.fit_transform(y_train_df.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.target_scaler.transform(y_test_df.values.reshape(-1, 1)).ravel()
        y_train_df = pd.Series(y_train_scaled, index=y_train_df.index, name=y_train_df.name)
        y_test_df = pd.Series(y_test_scaled, index=y_test_df.index, name=y_test_df.name)
        return y_train_df, y_test_df

    def preprocess_categorical_input(self, X_train_df, X_test_df, categorical_cols):
        # X_train_df = pd.get_dummies(X_train_df, columns=categorical_cols, drop_first=True) # turns categorical training data columns into one-hot encoded
        # X_test_df = pd.get_dummies(X_test_df, columns=categorical_cols, drop_first=True) # turns categorical testing data columns into one-hot encoded
        # X_train_df, X_test_df = X_train_df.align(X_test_df, join='left', axis=1, fill_value=0) # Align columns of train and test sets, filling missing columns with 0
        # return X_train_df, X_test_df
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(X_train_df[categorical_cols])
        encoded_train = pd.DataFrame(self.encoder.transform(X_train_df[categorical_cols]), index=X_train_df.index, columns=self.encoder.get_feature_names_out(categorical_cols))
        encoded_test = pd.DataFrame(self.encoder.transform(X_test_df[categorical_cols]), index=X_test_df.index, columns=self.encoder.get_feature_names_out(categorical_cols))
        X_train_df = X_train_df.drop(columns=categorical_cols).join(encoded_train)
        X_test_df = X_test_df.drop(columns=categorical_cols).join(encoded_test)
        return X_train_df, X_test_df
    
    def preprocess_categorical_output(self, y_train_df, y_test_df):
        # y_train_df = pd.get_dummies(y_train_df, drop_first=True)
        # y_test_df = pd.get_dummies(y_test_df, drop_first=True)
        # y_train_df, y_test_df = y_train_df.align(y_test_df, join='left', axis=1, fill_value=0)
        # return y_train_df, y_test_df
        num_classes = y_train_df.nunique()
        if num_classes <= 2:
            self.target_encoder = LabelEncoder()
            y_train_encoded = self.target_encoder.fit_transform(y_train_df)
            y_test_encoded = self.target_encoder.transform(y_test_df)
            y_train_df = pd.Series(y_train_encoded, index=y_train_df.index, name=self.target_column)
            y_test_df = pd.Series(y_test_encoded, index=y_test_df.index, name=self.target_column)
        else:
            self.target_encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
            y_train_encoded = self.target_encoder.fit_transform(y_train_df.values.reshape(-1, 1))
            y_test_encoded = self.target_encoder.transform(y_test_df.values.reshape(-1, 1))
            y_train_df = pd.DataFrame(y_train_encoded, index=y_train_df.index, columns=self.target_encoder.get_feature_names_out([self.target_column]))
            y_test_df = pd.DataFrame(y_test_encoded, index=y_test_df.index, columns=self.target_encoder.get_feature_names_out([self.target_column]))
        return y_train_df, y_test_df

    def preprocess(self, data):
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        df = df.dropna(subset=[self.target_column])

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if self.lstm_preprocessing:
            X_train_df, X_test_df, y_train_df, y_test_df = self.sequential_split(X, y)
        else:
            X_train_df, X_test_df, y_train_df, y_test_df = self.shuffle_split(X, y)


        if numerical_cols:
            X_train_df, X_test_df = self.preprocess_numerical_input(X_train_df, X_test_df, numerical_cols)
        if categorical_cols:
            X_train_df, X_test_df = self.preprocess_categorical_input(X_train_df, X_test_df, categorical_cols)
        
        if pd.api.types.is_numeric_dtype(y_train_df):
            y_train_df, y_test_df = self.preprocess_numerical_output(y_train_df, y_test_df)
        else:
            y_train_df, y_test_df = self.preprocess_categorical_output(y_train_df, y_test_df)

        if self.lstm_preprocessing:
            X_train_seq, y_train_seq = self.create_sequences(X_train_df, y_train_df)
            X_test_seq, y_test_seq = self.create_sequences(X_test_df, y_test_df)
        else:
            X_train_seq, y_train_seq = X_train_df.values, y_train_df.values
            X_test_seq, y_test_seq = X_test_df.values, y_test_df.values

        X_train_seq = X_train_seq.astype('float32')
        X_test_seq  = X_test_seq.astype('float32')
        y_train_seq = y_train_seq.astype('float32')
        y_test_seq  = y_test_seq.astype('float32')

        processed_data = {
            'X_train': X_train_seq,
            'X_test': X_test_seq,
            'y_train': y_train_seq,
            'y_test': y_test_seq,
            'input_scaler': self.scaler,
            'target_scaler': self.target_scaler
        }

        return processed_data