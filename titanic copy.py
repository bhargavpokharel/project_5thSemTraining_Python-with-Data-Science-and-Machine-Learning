{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c7605372",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c3a0a2d4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   PassengerId  Survived  Pclass  \\\n",
      "0            1         0       3   \n",
      "1            2         1       1   \n",
      "2            3         1       3   \n",
      "3            4         1       1   \n",
      "4            5         0       3   \n",
      "\n",
      "                                                Name     Sex   Age  SibSp  \\\n",
      "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
      "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
      "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
      "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
      "4                           Allen, Mr. William Henry    male  35.0      0   \n",
      "\n",
      "   Parch            Ticket     Fare Cabin Embarked  \n",
      "0      0         A/5 21171   7.2500   NaN        S  \n",
      "1      0          PC 17599  71.2833   C85        C  \n",
      "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
      "3      0            113803  53.1000  C123        S  \n",
      "4      0            373450   8.0500   NaN        S  \n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "# reading the data\n",
    "\n",
    "import pandas as pd\n",
    "\n",
    "train_df = pd.read_csv('train.csv')\n",
    "test_df = pd.read_csv('test.csv')\n",
    "\n",
    "print(train_df.head())\n",
    "print(train_df.info())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7761ac73",
   "metadata": {},
   "source": [
    "Age        → 714 non-null (missing ~177)\n",
    "Cabin      → Only 204 non-null (too many missing → drop)\n",
    "Embarked   → 889 non-null (missing 2 rows)\n",
    "\n",
    "Task:\n",
    "\n",
    "Fill Age with median (numeric).\n",
    "\n",
    "Drop Cabin (too many nulls, not worth fixing).\n",
    "\n",
    "Fill Embarked with mode (most frequent port).\n",
    "\n",
    "Drop Ticket and Name (not useful for now)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9f02fd21",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PassengerId    0\n",
      "Survived       0\n",
      "Pclass         0\n",
      "Sex            0\n",
      "Age            0\n",
      "SibSp          0\n",
      "Parch          0\n",
      "Fare           0\n",
      "Embarked       0\n",
      "dtype: int64\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Bhargav Pokharel\\AppData\\Local\\Temp\\ipykernel_24048\\3375003939.py:5: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  train_df[\"Age\"].fillna(train_df[\"Age\"].median(), inplace=True)\n",
      "C:\\Users\\Bhargav Pokharel\\AppData\\Local\\Temp\\ipykernel_24048\\3375003939.py:8: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  train_df[\"Embarked\"].fillna(train_df[\"Embarked\"].mode()[0], inplace=True)\n"
     ]
    }
   ],
   "source": [
    "# Drop columns with too many missing values or not useful\n",
    "train_df = train_df.drop(columns=[\"Cabin\", \"Ticket\", \"Name\"])\n",
    "\n",
    "# Fill missing Age with median\n",
    "train_df[\"Age\"].fillna(train_df[\"Age\"].median(), inplace=True)\n",
    "\n",
    "# Fill missing Embarked with mode\n",
    "train_df[\"Embarked\"].fillna(train_df[\"Embarked\"].mode()[0], inplace=True)\n",
    "\n",
    "# Confirm no more missing values and show how many missing values exist in each column\n",
    "print(train_df.isnull().sum())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3689b379",
   "metadata": {},
   "source": [
    "### Step 1: Encode Categorical Variables\n",
    "\n",
    "To prepare the data for machine learning models, we need to convert non-numeric (categorical) columns into numeric values:\n",
    "\n",
    "- Convert \"Sex\":\n",
    "  - male → 0\n",
    "  - female → 1\n",
    "\n",
    "- Convert \"Embarked\":\n",
    "  - S → 0\n",
    "  - C → 1\n",
    "  - Q → 2\n",
    "\n",
    "These encodings allow models like Random Forest to interpret the data correctly.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e6d64464",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_df[\"Sex\"] = train_df[\"Sex\"].map({\"male\": 0, \"female\": 1})\n",
    "train_df[\"Embarked\"] = train_df[\"Embarked\"].map({\"S\": 0, \"C\": 1, \"Q\": 2})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "862f1e59",
   "metadata": {},
   "source": [
    "### Step 2: Prepare Features and Target\n",
    "\n",
    "We separate the dataset into:\n",
    "\n",
    "- **Features (X)**: The columns that the model will use to learn and predict (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`)\n",
    "- **Target (y)**: The column to predict, which is `Survived` (0 = did not survive, 1 = survived)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "077272fe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Features shape: (891, 7)\n",
      "Target shape: (891,)\n",
      "\n",
      "Features used:\n",
      "['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']\n"
     ]
    }
   ],
   "source": [
    "# Cell 7: Prepare Features and Target\n",
    "# Separate features (X) and target variable (y)\n",
    "X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]\n",
    "y = train_df['Survived']\n",
    "\n",
    "print(\"Features shape:\", X.shape)\n",
    "print(\"Target shape:\", y.shape)\n",
    "print(\"\\nFeatures used:\")\n",
    "print(X.columns.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d0c70dab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set size: (712, 7)\n",
      "Validation set size: (179, 7)\n",
      "\n",
      "Data scaled successfully!\n"
     ]
    }
   ],
   "source": [
    "# Cell 8: Data Splitting and Scaling\n",
    "# Split the training data into training and validation sets\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Split data into training and validation sets\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "print(\"Training set size:\", X_train.shape)\n",
    "print(\"Validation set size:\", X_val.shape)\n",
    "\n",
    "# Scale numerical features\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_val_scaled = scaler.transform(X_val)\n",
    "\n",
    "print(\"\\nData scaled successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f719bd4f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 0.8156\n",
      "\n",
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.87      0.85       110\n",
      "           1       0.78      0.72      0.75        69\n",
      "\n",
      "    accuracy                           0.82       179\n",
      "   macro avg       0.81      0.80      0.80       179\n",
      "weighted avg       0.81      0.82      0.81       179\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Cell 9: Model Selection and Training\n",
    "# Train Random Forest classifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "\n",
    "# Create and train Random Forest model\n",
    "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Make predictions\n",
    "y_pred = rf_model.predict(X_val_scaled)\n",
    "\n",
    "# Evaluate model\n",
    "accuracy = accuracy_score(y_val, y_pred)\n",
    "print(f\"Random Forest Accuracy: {accuracy:.4f}\")\n",
    "\n",
    "# Display detailed classification report\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(y_val, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "7473dc3b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best parameters: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}\n",
      "Best cross-validation score: 0.8301\n",
      "Best validation accuracy: 0.8045\n"
     ]
    }
   ],
   "source": [
    "# Cell 10: Hyperparameter Tuning\n",
    "# Tune hyperparameters to improve accuracy\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "# Define parameter grid for Random Forest\n",
    "param_grid = {\n",
    "    'n_estimators': [50, 100, 200],\n",
    "    'max_depth': [None, 10, 20, 30],\n",
    "    'min_samples_split': [2, 5, 10],\n",
    "    'min_samples_leaf': [1, 2, 4]\n",
    "}\n",
    "\n",
    "# Perform grid search\n",
    "grid_search = GridSearchCV(RandomForestClassifier(random_state=42), \n",
    "                          param_grid, cv=3, scoring='accuracy', n_jobs=-1)\n",
    "grid_search.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Get best model\n",
    "best_model = grid_search.best_estimator_\n",
    "best_accuracy = accuracy_score(y_val, best_model.predict(X_val_scaled))\n",
    "\n",
    "print(f\"Best parameters: {grid_search.best_params_}\")\n",
    "print(f\"Best cross-validation score: {grid_search.best_score_:.4f}\")\n",
    "print(f\"Best validation accuracy: {best_accuracy:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e3ebfab9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test predictions completed!\n",
      "Predicted survivors: 136\n",
      "Predicted non-survivors: 282\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Bhargav Pokharel\\AppData\\Local\\Temp\\ipykernel_24048\\4230223748.py:9: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test_df_clean['Age'].fillna(test_df_clean['Age'].median(), inplace=True)\n",
      "C:\\Users\\Bhargav Pokharel\\AppData\\Local\\Temp\\ipykernel_24048\\4230223748.py:10: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test_df_clean['Fare'].fillna(test_df_clean['Fare'].median(), inplace=True)\n",
      "C:\\Users\\Bhargav Pokharel\\AppData\\Local\\Temp\\ipykernel_24048\\4230223748.py:11: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test_df_clean['Embarked'].fillna(test_df_clean['Embarked'].mode()[0], inplace=True)\n"
     ]
    }
   ],
   "source": [
    "# Cell 11: Predict on Test Data\n",
    "# Apply the selected model on the test dataset\n",
    "# First, clean and preprocess test data similar to training data\n",
    "\n",
    "# Clean test data\n",
    "test_df_clean = test_df.drop(columns=['Cabin', 'Ticket', 'Name'])\n",
    "\n",
    "# Fill missing values in test data\n",
    "test_df_clean['Age'].fillna(test_df_clean['Age'].median(), inplace=True)\n",
    "test_df_clean['Fare'].fillna(test_df_clean['Fare'].median(), inplace=True)\n",
    "test_df_clean['Embarked'].fillna(test_df_clean['Embarked'].mode()[0], inplace=True)\n",
    "\n",
    "# Encode categorical variables\n",
    "test_df_clean['Sex'] = test_df_clean['Sex'].map({'male': 0, 'female': 1})\n",
    "test_df_clean['Embarked'] = test_df_clean['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})\n",
    "\n",
    "# Prepare test features\n",
    "X_test = test_df_clean[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]\n",
    "\n",
    "# Scale test features\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "# Make predictions\n",
    "test_predictions = best_model.predict(X_test_scaled)\n",
    "\n",
    "# Create submission dataframe\n",
    "submission = pd.DataFrame({\n",
    "    'PassengerId': test_df['PassengerId'],\n",
    "    'Survived': test_predictions\n",
    "})\n",
    "\n",
    "print(\"Test predictions completed!\")\n",
    "print(f\"Predicted survivors: {sum(test_predictions)}\")\n",
    "print(f\"Predicted non-survivors: {len(test_predictions) - sum(test_predictions)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "1ab3531f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions saved to 'titanic_predictions.csv'\n",
      "\n",
      "First 10 predictions:\n",
      "   PassengerId  Survived\n",
      "0          892         0\n",
      "1          893         0\n",
      "2          894         0\n",
      "3          895         0\n",
      "4          896         0\n",
      "5          897         0\n",
      "6          898         1\n",
      "7          899         0\n",
      "8          900         1\n",
      "9          901         0\n"
     ]
    }
   ],
   "source": [
    "# Cell 12: Save Predictions\n",
    "# Save predictions to CSV file\n",
    "submission.to_csv('titanic_predictions.csv', index=False)\n",
    "print(\"Predictions saved to 'titanic_predictions.csv'\")\n",
    "print(\"\\nFirst 10 predictions:\")\n",
    "print(submission.head(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "10e6d4f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature Importance:\n",
      "    feature  importance\n",
      "1       Sex    0.353252\n",
      "5      Fare    0.214589\n",
      "2       Age    0.183784\n",
      "0    Pclass    0.130214\n",
      "3     SibSp    0.046108\n",
      "6  Embarked    0.036872\n",
      "4     Parch    0.035181\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABdIAAASlCAYAAACspitqAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjMsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvZiW1igAAAAlwSFlzAAAPYQAAD2EBqD+naQAA5s5JREFUeJzs3QeYVNX5OP5DkS6gooAGwV5BsHfEhmKJGkuMBbHEHg0alVgQG/ZeY9fYNRp7jRgLiYoaOzYIRqVYAEUFhfk/7/n9Z7+7yzKC7jIL+/k8z7g7996599wzw3ruO+99T6NCoVBIAAAAAABAjRrXvBgAAAAAAAgC6QAAAAAAUIJAOgAAAAAAlCCQDgAAAAAAJQikAwAAAABACQLpAAAAAABQgkA6AAAAAACUIJAOAAAAAAAlCKQDAAAAAEAJAukAAABAg7DPPvukbt261ekxGjVqlE4++eQ0P4k+23bbbVNDNjc+O0D9JpAO8AvdcMMNebBc0+O4446rk2O+8MILeXA+ceLEVF/74+WXX07zqssvvzyfBwAAP98bb7yRdt5559S1a9fUokWLtMQSS6QtttgiXXLJJamhGz16dJXrhsaNG6eFF144bb311mn48OENehw7efLkdPrpp6c111wztWvXLjVv3jx/hnbbbbf00EMPlbt5QAPWtNwNAJhfnHLKKWmppZaqsmzVVVets0D6kCFDclZE+/bt6+QYDVlcgHTo0CH3LwAAP2+82qdPn7TkkkumAw44IHXq1Cl9/PHH6V//+le66KKL0uGHH16Wdl199dVpxowZqb7YfffdU79+/dL06dPTe++9l8eh0W8vvfRS6t69e4Mbx37wwQepb9++6b///W/acccd0957753atGmTPzsPP/xwzoq/6aab0l577VXupgINkEA6QC2J7JHImpiXTZkyJbVu3To1VN9++21q1apVuZsBADDPi4ziyCaOgHD1xI/x48eXbfy6wAILpPpk9dVXT3vuuWfF84022ihfV1xxxRU5KN6Q/Pjjjzl4Pm7cuPTMM8+kDTbYoMr6wYMHp8cffzx/6TCvKxQK6fvvv08tW7Ysd1OAOaC0C8Bc8sgjj+SBcQz0F1xwwbTNNtukt956q8o2r7/+es4eWXrppfPtr5G5s++++6YvvviiYpso6fKnP/0p/x4Z8MXbQeP20OItojXdzlm9VmP8Hsvefvvt9Lvf/S4ttNBCacMNN6xY/9e//jWtscYaeXAXt5n+9re/zZkgP0ecU2SSjBkzJmeRxO9xa+9ll11Wcdvvpptumvsmbtu89dZbaywX889//jMdeOCBaZFFFklt27bNGSpfffXVTMeLi45VVlkl3wa6+OKLp0MPPXSmMjibbLJJvmNgxIgRaeONN84B9D//+c+57mG8LzF4L/ZtbBu+/PLLdPTRR+fsoDiHaENc6PznP/+psu9hw4bl19155535IvJXv/pVfj8322yznGVT3b///e+ciRTvQfRBjx49cqZWZe+++26+NTrei9hXfGlz//33/6z3AwCgrn344Yd5PFbT3ZOLLbZYxe+1MX4999xz8/LIYq5u0KBBqVmzZhVjxsp1rn/44Yc8thowYECN5UVizBVjvzBt2rR00kkn5fFxfEEQY7YY2z/99NOpNsU+i/1X2fXXX5/Hy9F3McZdeeWVc7C9slLj2BDj4SOPPDJ16dIl72PZZZdNZ5111hxl6Ecgu2fPnrlvog1/+9vfKtZ99NFH+ZgXXHBBjXcoxLrbbrttlvu+66670ptvvplOPPHEmYLoRVtuuWUef1c2O+dV/JzFZ+Uvf/lLWmaZZfK2a621Vv6yp7r77rsvXyvEecbPe++9t8b2xDEuvPDC/FmPbTt27JivV6pfoxRrzD/22GN5HB/XWFddddUs+wKon2SkA9SSSZMmpc8//7zKsritMtx8882pf//++TbFGNRF5nMMfGPg/+qrr1YM5p944ok8AI3BfATRYyAcA734GbfBxuBvp512yrd9xiA0BqnFYyy66KJpwoQJc9zuXXbZJS233HLpjDPOyJkRIYK/MYDddddd0/7775/3G7UsI+Ac7f055WQicyQGvbGPs88+O91yyy3psMMOyxchxx9/fNpjjz3yuV155ZU5QL7eeuvNVConto9jx0XUyJEjcx/GBVMxcB1iXZS92XzzzdPBBx9csV0MkJ9//vkqWUjxBUW0Kb4kiEygGPjGxUbcahyB8mhXiOUh3psYVEefRdsiWyYGwL17984XdBG0r+zMM8/M9S7jAiw+H3HecZ4ROC+K9zwG1Z07d05HHHFEft/feeed9OCDD+bnId7/uJiILx+i7n70WQTpd9hhh3TPPffkzB0AgPokkiOi1ncERmu73GH18WuMpY455pg8PiomnBTF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ri4msClmrMT+Ils5sm1i4BwTep555pn5Z2T8xEVJMVNkdkRmxWmnnZYGDRqUax3G7atRrzDaEQPnqOMYZUbKITLL47bWGLRGRkhcYEXWUExiFaJfo93xJUBkwsTy4nZxK/FPTXBUuX/jPYt+iLIpEcyOmo5xMXbKKafk25HXX3/99MYbb+SMmcrZ73MiLtriOPHexYA99huD96i7+dZbb1VcEMREtnGecRESE6fG8caNG5cvGv73v//lmqMAAPXJ4YcfnjORY3LKKNkR47go9RFZ35G9XHnC9186fi2KMVvcPRjlT2KS9urB8FJi20suuSRnmMeYa6WVVqqyPsaBkY0e5xPJGTE2vvLKK3PgNzKia9MRRxyREy2iTyJDe8stt8zJIjFmjCB7HC+y1uN8qydazGocG+VoIqM9ziNK1MR2U6ZMyePZyLaPcf9P3WEZd6vut99+6aWXXkodO3ZM1113XR6T1hTQj/IukdwSX6jMTnmdENc0cb0RiUgx9o0EnyjXE9c4kbgS7Y9rqMrJMbVxXtVFuZ44RrQhSjrGPFbx2VhllVWqvNcx2Wu8H7F9fOkR71OcQ2Szxxc5MbdS9UljgXlcAYBf5Prrr4+0iMJLL71Ucrunn3660Ldv30K7du0KLVq0KCyzzDKFffbZp/Dyyy9XbPO///2vsOOOOxbat2+ft9tll10Kn376ad7/4MGDq+zv1FNPLSyxxBKFxo0b5/WjRo3Ky7/99tvCfvvtl1+/4IILFnbdddfC+PHjZ9pH/B7LJkyYUGN777nnnsKGG25YaN26dX6suOKKhUMPPbQwcuTIOe6P/v37531U17t378Iqq6wy0/KuXbsWttlmm5n2+cwzzxR+//vfFxZaaKFCmzZtCnvssUfhiy++mOn1l156aW7vAgssUOjYsWPh4IMPLnz11VezdewwduzYfPzovzhubBu+//77wlFHHVXo3LlzoWXLloUNNtigMHz48Ly+uE3xvY7X3XXXXVX2G+9RLI/zqey5554rbLHFFvl40U89evQoXHLJJVW2+fDDDwt77713oVOnTvm84r3fdtttC3fffXeN5wAAUE6PPPJIYd99981jshi3NWvWrLDssssWDj/88MK4ceOqbFtb49dw9dVX521iP999991M62NcGmPN6mbMmFHo0qVLfu1pp51W4/ozzjgjv7Z58+aFXr16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TIMHD06vvPJKWm211VLfvn3T+PHja9z+1ltvTccdd1ze/p133knXXntt3sef//znud52AACoiUA6AABQq84///x0wAEHpAEDBqSVV145XXnllalVq1bpuuuuq3H7F154IW2wwQbpd7/7Xc5i33LLLdPuu+/+k1nsAAAwtwikAwAAtWbatGlpxIgRafPNN69Y1rhx4/x8+PDhNb5m/fXXz68pBs4/+uij9PDDD6d+/frVuP3UqVPT5MmTqzwAAKAuNa3TvQMAAA3K559/nqZPn546duxYZXk8f/fdd2t8TWSix+s23HDDVCgU0o8//pgOOuigWZZ2GTp0aBoyZEidtB8AAGoiIx0AACirYcOGpTPOOCNdfvnluab63/72t/TQQw+lU089tcbtBw0alCZNmlTx+Pjjj+d6mwEAaFhkpAMAALWmQ4cOqUmTJmncuHFVlsfzTp061fiaE088Me21115p//33z8+7d++epkyZkn7/+9+n448/PpeGqax58+b5AQAAc4uMdAAAoNY0a9YsrbHGGumpp56qWDZjxoz8fL311qvxNd9+++1MwfIIxoco9QIAAOUmIx0AAKhVAwcOTP37909rrrlmWnvttdOFF16YM8wHDBiQ1++9995piSWWyLXOw3bbbZfOP//81KtXr7TOOuukDz74IGepx/JiQB0AAMpJIB0AAKhVu+22W5owYUI66aST0tixY1PPnj3To48+WjEB6ZgxY6pkoJ9wwgmpUaNG+ecnn3ySFl100RxEP/3008t4FgAA8H8aFdwrWecmT56c2rVrl7oceWdq3LxVashGn7lNuZsAADDPjSNjQs22bduWuzn1ln4CAKCux5FqpAMAAAAAQAkC6QAAAAAAUIJAOgAAAAAAlCCQDgAAAAAAJQikAwAAAABAfQ6kn3zyyalnz551su9hw4alRo0apYkTJ9baPkePHp33+dprr9XaPgEAAAAAmE8C6fvss08OIld/bLXVVnXXQgAAAAAAKKOmc/qCCJpff/31VZY1b9481Tc//PBDuZsAAAAAAEBDLO0SQfNOnTpVeSy00EJ5XWSnX3XVVWnbbbdNrVq1SiuttFIaPnx4+uCDD9Imm2ySWrdundZff/304YcfzrTfeF2XLl3y63bdddc0adKkinUvvfRS2mKLLVKHDh1Su3btUu/evdMrr7xS5fVx7CuuuCJtv/32+Tinn376TMf49ttv09Zbb5022GCDinIv11xzTW5nixYt0oorrpguv/zyKq958cUXU69evfL6NddcM7366qs/2UdTp05NkydPrvIAAAAAAGDeVOs10k899dS099575xriEZj+3e9+lw488MA0aNCg9PLLL6dCoZAOO+ywKq+JQPudd96ZHnjggfToo4/mYPUhhxxSsf7rr79O/fv3T88991z617/+lZZbbrnUr1+/vLx6vfUdd9wxvfHGG2nfffetsi4C5xGMnzFjRnriiSdS+/bt0y233JJOOumkHHR/55130hlnnJFOPPHEdOONN+bXfPPNN/lLgZVXXjmNGDEi7//oo4/+yT4YOnRoDvgXH/EFAQAAAAAADSSQ/uCDD6Y2bdpUeUQAumjAgAE5o3z55ZdPxx57bJ6cc4899kh9+/bNmd9HHHFEngS0su+//z7ddNNNedLRjTfeOF1yySXp9ttvT2PHjs3rN91007TnnnvmwHzs4y9/+UvOLn/mmWeq7CeC9nH8pZdeOi255JIVy2M/kcXeuXPnHKyPrPcwePDgdN5556WddtopLbXUUvnnH//4x5wdH2699dYceL/22mvTKquskoPqf/rTn36yj+JLg8ioLz4+/vjjOe1mAAAAAADm1Rrpffr0ySVUKlt44YUrfu/Ro0fF7x07dsw/u3fvXmVZBM6j3Enbtm3zsgh6L7HEEhXbrLfeejmAPXLkyFw6Zty4cemEE07IAfjx48en6dOn50D6mDFjqrQjSq/UJDLR11577XTHHXekJk2a5GVTpkzJJWb222+/dMABB1Rs++OPP+Ys8hBZ6nE+Udalcttmp/xNfawbDwAAAADAXAikR/3xZZdddpbrF1hggSp1y2e1LALlsyvKunzxxRfpoosuSl27ds1B6ghoT5s2baa21WSbbbZJ99xzT3r77bcrgvpRtiVcffXVaZ111qmyfTHYDgAAAAAAcxxIrwuRWf7pp5+mxRdfPD+POuiNGzdOK6ywQn7+/PPP50lAoy56iFIpn3/++Wzv/8wzz8wlaDbbbLOc1R41zyMzPo730Ucf5dIzNYkyMjfffHPOoC9mpUfbAAAAAABoOOY4kD516tSK2uUVO2naNHXo0OFnNyKC1JF1fu655+aSL3/4wx9ynfUo6xJictEIaEfpllgfdcpbtmw5R8eIfUdJmKi3HsH0qLc+ZMiQfKwo5bLVVlvlc4sJUb/66qs0cODAXHP9+OOPz6Vfou551HuP/QAAAAAA0HDM8WSjjz76aJ60s/Jjww03/EWNiFIxMdFnZJxvueWWuS55ZKAXxWSfEdxeffXV01577ZWD34stttgcH+eCCy7IAfoIpr/33ntp//33T9dcc026/vrrc8mXmJD0hhtuyBOPhshij8lJ33jjjdSrV68cVD/rrLN+0bkCAAAAADBvaVQoFArlbsT8LrLoI+u9y5F3psbNW6WGbPSZ25S7CQAA89w4ctKkSalt27blbk69pZ8AAKjrceQcZ6QDAAAAAEBDIpAOAAAAAAAlCKQDAAAAAEAJAukAAAAAAFCCQDoAAAAAAJTQtNRKAAAAAPilJg0ZUu4mAPVcu8GDU30mIx0AAAAAAEoQSAcAAAAAgBIE0gEAAAAAoASBdAAAAAAAKEEgHQAAAAAAShBIBwAAAACAEgTSAQAAAACgBIF0AAAAAAAoQSAdAAAAAABKEEgHAAAAAIASBNIBAAAAAKAEgXQAAAAAAChBIB0AAAAAAEoQSAcAAAAAgBKallpJ7XpzSN/Utm3bcjcDAAAAAIA5ICMdAAAAAABKEEgHAAAAAIASBNIBAAAAAKAEgXQAAAAAAChBIB0AAAAAAEoQSAcAAAAAgBIE0gEAAAAAoASBdAAAAAAAKEEgHQAAAAAAShBIBwAAAACAEgTSAQAAAACgBIF0AAAAAAAooWmpldSuVQc/lho3b1XuZszTRp+5TbmbAAAAAAA0MDLSAQAAAACgBIF0AAAAAAAoQSAdAAAAAABKEEgHAAAAAIASBNIBAAAAAKAEgXQAAAAAAChBIB0AAAAAAEoQSAcAAAAAgBIE0gEAAAAAoASBdAAAAAAAKEEgHQAAAAAAShBIBwAAAACAEgTSAQAAAACgBIF0AAAAAAAoQSAdAAAAAABKEEgHAAAAAIASBNIBAAAAAKAEgXQAAAAAAChBIB0AAAAAAEoQSAcAAAAAgBIE0gEAAAAAoASBdAAAAAAAaCiB9EaNGqX77rsv/z569Oj8/LXXXit3swAAAAAAmIfNU4H0CRMmpIMPPjgtueSSqXnz5qlTp06pb9++6fnnn8/rP/vss7T11lvP0T7vvffetO6666Z27dqlBRdcMK2yyirpyCOPrKMzAAAAAABgXtM0zUN+85vfpGnTpqUbb7wxLb300mncuHHpqaeeSl988UVeH4H1ORGv3W233dLpp5+ett9++5zB/vbbb6cnnniijs4AAAAAAIB5zTyTkT5x4sT07LPPprPOOiv16dMnde3aNa299tpp0KBBOQhevbRL0bvvvpvWX3/91KJFi7TqqqumZ555pmLdAw88kDbYYIP0pz/9Ka2wwgpp+eWXTzvssEO67LLLKrY5+eSTU8+ePdNVV12VunTpklq1apV23XXXNGnSpLl49gAAAAAAlMs8E0hv06ZNfkSgfOrUqbP9ugiSH3XUUenVV19N6623Xtpuu+2qZLC/9dZb6c033yy5jw8++CDdeeedOfD+6KOP5n0dcsghs9w+2jd58uQqDwAAAAAA5k3zTCC9adOm6YYbbshlXdq3b58zyf/85z+n119/veTrDjvssFwSZqWVVkpXXHFFroV+7bXX5nWHH354WmuttVL37t1Tt27d0m9/+9t03XXXzRSo//7779NNN92UM9M33njjdMkll6Tbb789jR07tsZjDh06NB+n+IhMdgAAAAAA5k3zTCA9RED8008/Tffff3/aaqut0rBhw9Lqq6+eA+yzElnolYPxa665ZnrnnXfy89atW6eHHnooZ5yfcMIJOeM9stejZMy3335b8bqY3HSJJZaoss8ZM2akkSNH1njMKDcTpV+Kj48//riWegAAAAAAgLltngqkh6h1vsUWW6QTTzwxvfDCC2mfffZJgwcP/kX7XGaZZdL++++frrnmmvTKK6/kCUfvuOOOn72/5s2bp7Zt21Z5AAAAAAAwb5rnAunVrbzyymnKlCmzXP+vf/2r4vcff/wxjRgxIpd5mZUo8RITilbe55gxY3ImfOV9Nm7cOE9QCgAAAADA/K1pmkfEBKG77LJL2nfffVOPHj3SggsumF5++eV09tlnp1//+tezfN1ll12WlltuuRw8v+CCC9JXX32V9xFOPvnkXMKlX79+qWvXrmnixInp4osvTj/88EPOeq+cBd+/f/907rnn5olD//CHP6Rdd901T1YKAAAAAMD8bZ4JpEf98nXWWScHwz/88MMc7I5JPA844IA86eisnHnmmfnx2muvpWWXXTbXV+/QoUNe17t37xxo33vvvdO4cePSQgstlHr16pUef/zxKtnm8bqddtopB9y//PLLtO2226bLL798rpw3AAAAAADlNc8E0qPu+NChQ/NjVgqFQpUSLcXnu+++e43b9+nTJz9mx8EHH5wfAAAAAAA0LPN8jXQAAAAAAKhLAukAAAAAAFCCQPpPiAlJo746AAAAAAANk0A6AAAAAACUIJAOAAAAAAAlCKQDAAAAAEAJAukAAAAAAFCCQDoAAFDrLrvsstStW7fUokWLtM4666QXX3yx5PYTJ05Mhx56aOrcuXNq3rx5Wn755dPDDz8819oLAAClNC25FgAAYA7dcccdaeDAgenKK6/MQfQLL7ww9e3bN40cOTIttthiM20/bdq0tMUWW+R1d999d1piiSXSf//739S+ffuytB8AAKoTSAcAAGrV+eefnw444IA0YMCA/DwC6g899FC67rrr0nHHHTfT9rH8yy+/TC+88EJaYIEF8rLIZp+VqVOn5kfR5MmT6+Q8AACgSGkXAACg1kR2+YgRI9Lmm29esaxx48b5+fDhw2t8zf3335/WW2+9XNqlY8eOadVVV01nnHFGmj59eo3bDx06NLVr167i0aVLlzo7HwAACALpAABArfn8889zADwC4pXF87Fjx9b4mo8++iiXdInXRV30E088MZ133nnptNNOq3H7QYMGpUmTJlU8Pv744zo5FwAAKFLaBQAAKKsZM2bk+uh/+ctfUpMmTdIaa6yRPvnkk3TOOeekwYMHz7R9TEYaDwAAmFsE0gEAgFrToUOHHAwfN25cleXxvFOnTjW+pnPnzrk2eryuaKWVVsoZ7FEqplmzZnXebgAAKEVpFwAAoNZE0Dsyyp966qkqGefxPOqg12SDDTZIH3zwQd6u6L333ssBdkF0AADqA4F0AACgVg0cODBdffXV6cYbb0zvvPNOOvjgg9OUKVPSgAED8vq999471zkvivVffvllOuKII3IA/aGHHsqTjcbkowAAUB8o7QIAANSq3XbbLU2YMCGddNJJuTxLz54906OPPloxAemYMWNS48b/l9PTpUuX9Nhjj6U//vGPqUePHmmJJZbIQfVjjz22jGcBAAD/RyAdAACodYcddlh+1GTYsGEzLYuyL//617/mQssAAGDOKe0CAACkZ599Nu255545oP3JJ5/kZTfffHN67rnnyt00AAAoO4F0AABo4O65557Ut2/f1LJly/Tqq6+mqVOn5uWTJk3KtcoBAKChU9plLnpzSN/Utm3bcjcDAACqOO2009KVV16ZJwG9/fbbK5ZvsMEGeR0AADR0MtIBAKCBGzlyZNp4441nWt6uXbs0ceLEsrQJAADqE4F0AABo4Dp16pQ++OCDmZZHffSll166LG0CAID6RCAdAAAauAMOOCAdccQR6d///ndq1KhR+vTTT9Mtt9ySjj766HTwwQeXu3kAAFB2aqQDAEADd9xxx6UZM2akzTbbLH377be5zEvz5s1zIP3www8vd/MAAKDsBNIBAKCBiyz0448/Pv3pT3/KJV6++eabtPLKK6c2bdqUu2kAAFAvKO0CAAAN3L777pu+/vrr1KxZsxxAX3vttXMQfcqUKXkdAAA0dALpAADQwN14443pu+++m2l5LLvpppvK0iYAAKhPlHYBAIAGavLkyalQKORHZKS3aNGiYt306dPTww8/nBZbbLGythEAAOoDgXQAAGig2rdvn+ujx2P55ZefaX0sHzJkSFnaBgAA9YlAOgAANFBPP/10zkbfdNNN0z333JMWXnjhinVRL71r165p8cUXL2sbAQCgPhBIBwCABqp3797556hRo1KXLl1S48amUAIAgJoIpAMAQAMXmefh22+/TWPGjEnTpk2rsr5Hjx5lahkAANQPAukAANDATZgwIQ0YMCA98sgjNa6PiUcBAKAhE0ifi1Yd/Fhq3LxVuZsBVDL6zG3K3QQAKLsjjzwyTZw4Mf373/9Om2yySbr33nvTuHHj0mmnnZbOO++8cjcPAADKTiAdAAAauH/84x/p73//e1pzzTVznfQo9bLFFluktm3bpqFDh6ZttvHFMwAADZvZhAAAoIGbMmVKWmyxxfLvCy20UC71Erp3755eeeWVMrcOAADKTyAdAAAauBVWWCGNHDky/77aaqulq666Kn3yySfpyiuvTJ07dy538wAAoOyUdgEAgAbuiCOOSJ999ln+ffDgwWmrrbZKt9xyS2rWrFm64YYbyt08AAAoO4F0AABo4Pbcc8+K39dYY4303//+N7377rtpySWXTB06dChr2wAAoD5Q2gUAAKiiVatWafXVV09t2rRJ5557brmbAwAAZSeQDgAADVhMLPrggw+mxx9/PE2fPj0v++GHH9JFF12UunXrls4888xyNxEAAMpOaRcAAGignnvuubTtttumyZMnp0aNGqU111wzXX/99WmHHXZITZs2TSeffHLq379/uZsJAABlJyMdAAAaqBNOOCH169cvvf7662ngwIHppZdeSjvuuGM644wz0ttvv50OOuig1LJly3I3EwAAyk4gHQAAGqg33ngjB9NXXXXVdMopp+Ss9LPPPjvtvPPO5W4aAADUKwLpAADQQH311VepQ4cO+ffIPI9JRiOoDgAAVKVGOgAANGBRwmXs2LH590KhkEaOHJmmTJlSZZsePXqUqXUAAFA/CKQDAEADttlmm+UAelFMPhqizEssj5/Tp08vYwsBAKD8BNIBAKCBGjVqVLmbAAAA8wSBdAAAaKC6du1a7iYAAMA8wWSjAAAAAABQgkA6AAAAAACUIJAOAAAAAAAlCKQDAAAAAEAJAukAAAAAAFBC01IrAQCA+VOvXr1So0aNZmvbV155pc7bAwAA9ZlAOgAANEA77LBDuZsAAADzjPk2kL7JJpuknj17pgsvvLDcTQEAgHpn8ODB5W4CAADMM+p1jfR99tkn324aj2bNmqVll102nXLKKenHH38sd9MAAAAAAGgg6n1G+lZbbZWuv/76NHXq1PTwww+nQw89NC2wwAJp0KBB5W4aAADMF6ZPn54uuOCCdOedd6YxY8akadOmVVn/5Zdflq1tAABQH9TrjPTQvHnz1KlTp9S1a9d08MEHp8033zzdf//9ed3zzz+fS7i0atUqLbTQQqlv377pq6++qnE/N998c1pzzTXTggsumPf3u9/9Lo0fP75ifbxujz32SIsuumhq2bJlWm655XIAP8SFxGGHHZY6d+6cWrRokdsydOjQWbY5gv6TJ0+u8gAAgPpqyJAh6fzzz0+77bZbmjRpUho4cGDaaaedUuPGjdPJJ59c7uYBAEDZ1ftAenUR5I7A9muvvZY222yztPLKK6fhw4en5557Lm233XY5m6YmP/zwQzr11FPTf/7zn3Tfffel0aNH59IxRSeeeGJ6++230yOPPJLeeeeddMUVV6QOHTrkdRdffHEO3keGzsiRI9Mtt9ySunXrNss2RpC9Xbt2FY8uXbrUQU8AAEDtiPHt1VdfnY466qjUtGnTtPvuu6drrrkmnXTSSelf//pXuZsHAABlV+9LuxQVCoX01FNPpcceeywdfvjh6eyzz84Z5pdffnnFNqusssosX7/vvvtW/L700kvn4Phaa62Vvvnmm9SmTZt8C2uvXr3yPkPlQHmsiwz1DTfcMNdrj4z0UqLsTGTxFEVGumA6AAD11dixY1P37t3z7zE2jqz0sO222+aEEwAAaOjqfUb6gw8+mAfzUVJl6623zrebxu2lxYz02TVixIicsb7kkkvm8i69e/euCJKHKBtz++23p549e6ZjjjkmvfDCCxWvjcz1ON4KK6yQ/vCHP6THH3/8J8vRtG3btsoDAADqq1/96lfps88+y78vs8wyFePdl156KY9tAQCgoav3gfQ+ffrkIPb777+fvvvuu3TjjTem1q1b5xIvs2vKlCm5fnoEtOO21bgguPfee/O64kRKEaT/73//m/74xz+mTz/9NAfpjz766Lxu9dVXT6NGjcqlYaINu+66a9p5553r6IwBAGDu2nHHHfPdnyHu/ows9Lgjc++9965yZycAADRU9b60SwTNl1122ZmW9+jRIw/2Y2Kkn/Luu++mL774Ip155pkVJVZefvnlmbaLiUb79++fHxtttFH605/+lM4999y8LoLwkQ0fjwiib7XVVunLL79MCy+8cK2cJwAAlEuMk4tivBulDOMOzQimx12dAADQ0NX7QHqpOuRRx/GQQw5JBx10UGrWrFl6+umn0y677FIxSWhRlHOJ9Zdcckne9s0338zZ5ZXFREprrLFGrrM+derUXFJmpZVWyuvOP//81Llz51xDvXHjxumuu+5KnTp1Su3bt5+r5wwAAHXh+++/z6UUi9Zdd938AAAA5pHSLrOy/PLL59qN//nPf9Laa6+d1ltvvfT3v/89NW3atMZM8xtuuCEHwFdeeeWccVPMNC+KQHsE5yPTfeONN05NmjTJNdND1FQvTm4aE5SOHj06PfzwwzmoDgAA87rFFlss35X5xBNPpBkzZpS7OQAAUO80KhQKhXI3Yn43efLk1K5du9TlyDtT4+atyt0coJLRZ25T7iYAwE+OIydNmlSnE9jH/EG33npreuihh/LxorzLnnvumRNJ5gVzq58A+PkmzUZpXqBhazd4cL0eR0qpBgCABi4mG427N8eNG5fOOOOM9Pbbb+fSLnEX6CmnnFLu5gEAQNkJpAMAABUlDQcMGJBLKL7++uupdevWaYgMQgAAEEgHAAD+b9LRO++8M+2www5p9dVXT19++WX605/+VO5mAQBA2c08MycAANCgPPbYY7lG+n333ZeaNm2adt5555yVvvHGG5e7aQAAUC8IpAMAQAMXNdK33XbbdNNNN6V+/fqlBRZYoNxNAgCAekUgHQAAGriYZDTqowMAADUTSAcAgAZo8uTJqW3btvn3QqGQn89KcTsAAGioBNIBAKABWmihhdJnn32WFltssdS+ffvUqFGjmbaJAHssnz59elnaCAAA9YVAOgAANED/+Mc/0sILL1zxe02BdAAA4P8RSAcAgAaod+/eFb9vsskmZW0LAADUd43L3QAAAKC8lltuuXTyySen999/v9xNAQCAekkgHQAAGrhDDjkkPfTQQ2nFFVdMa621VrrooovS2LFjy90sAACoNwTSAQCggfvjH/+YXnrppfTOO++kfv36pcsuuyx16dIlbbnllummm24qd/MAAKDsBNIBAIBs+eWXT0OGDEnvvfdeevbZZ9OECRPSgAEDyt0sAAAoO5ONAgAAFV588cV06623pjvuuCNNnjw57bLLLuVuEgAAlJ1AOgAANHCRgX7LLbek2267LY0aNSptuumm6ayzzko77bRTatOmTbmbBwAAZSeQDgAADVxxktFDDz00/fa3v00dO3Ysd5MAAKBeEUgHAIAGbPr06emqq65KO++8c1pooYXK3RwAAKiXTDYKAAANWJMmTdLhhx+eJk6cWO6mAABAvSWQDgAADdyqq66aPvroo3I3AwAA6i2lXeaiN4f0TW3bti13MwAAoIrTTjstHX300enUU09Na6yxRmrdunWV9cawAAA0dALpAADQwPXr1y//3H777VOjRo0qlhcKhfw86qgDAEBDJpAOAAAN3NNPP13uJgAAQL0mkA4AAA1c7969y90EAACo1wTSAQCggfvnP/9Zcv3GG28819oCAAD1kUA6AAA0cJtssslMyyrXSlcjHQCAhq5xuRsAAACU11dffVXlMX78+PToo4+mtdZaKz3++OPlbh4AAJSdjHQAAGjg2rVrN9OyLbbYIjVr1iwNHDgwjRgxoiztAgCA+kJGOgAAUKOOHTumkSNHlrsZAABQdjLSAQCggXv99derPC8UCumzzz5LZ555ZurZs2fZ2gUAAPWFQDoAADRwESyPyUUjgF7Zuuuum6677rqytQsAAOoLgXQAAGjgRo0aVeV548aN06KLLppatGhRtjYBAEB9IpAOAAANXNeuXcvdBAAAqNcE0ueiVQc/lho3b1XuZgDUC6PP3KbcTQBo8IYPH56++OKLtO2221Ysu+mmm9LgwYPTlClT0g477JAuueSS1Lx587K2EwAAyq1xuRsAAACUxymnnJLeeuutiudvvPFG2m+//dLmm2+ejjvuuPTAAw+koUOHlrWNAABQHwikAwBAA/Xaa6+lzTbbrOL57bffntZZZ5109dVXp4EDB6aLL7443XnnnWVtIwAA1AcC6QAA0EB99dVXqWPHjhXPn3nmmbT11ltXPF9rrbXSxx9/XKbWAQBA/SGQDgAADVQE0UeNGpV/nzZtWnrllVfSuuuuW7H+66+/TgsssEAZWwgAAPWDQDoAADRQ/fr1y7XQn3322TRo0KDUqlWrtNFGG1Wsf/3119MyyyxT1jYCAEB90LTcDQAAAMrj1FNPTTvttFPq3bt3atOmTbrxxhtTs2bNKtZfd911acsttyxrGwEAoD6QkQ4AAA1Uhw4d0j//+c9cKz0eO+64Y5X1d911Vxo8ePDP2vdll12WunXrllq0aJEnMH3xxRdn63Ux4WmjRo3SDjvs8LOOCwAAdUEgHQAAGrh27dqlJk2azLR84YUXrpKhPrvuuOOONHDgwByEj7rrq622Wurbt28aP358ydeNHj06HX300VXKywAAQH0gkA4AANSq888/Px1wwAFpwIABaeWVV05XXnllrr8epWJmZfr06WmPPfZIQ4YMSUsvvXTJ/U+dOjVNnjy5ygMAAOqSQDoAAFBrpk2blkaMGJE233zzimWNGzfOz4cPHz7L151yyilpscUWS/vtt99PHmPo0KE5i7746NKlS621HwAAaiKQDgAA1JrPP/88Z5d37NixyvJ4Pnbs2Bpf89xzz6Vrr702XX311bN1jEGDBqVJkyZVPD7++ONaaTsAAMxK01muAQAAqGNff/112muvvXIQPSY/nR3NmzfPDwAAmFsE0gEAgFoTwfCYuHTcuHFVlsfzTp06zbT9hx9+mCcZ3W677SqWzZgxI/9s2rRpGjlyZFpmmWXmQssBAGDWlHYBAABqTbNmzdIaa6yRnnrqqSqB8Xi+3nrrzbT9iiuumN5444302muvVTy233771KdPn/y7+ucAANQHMtIBAIBaNXDgwNS/f/+05pprprXXXjtdeOGFacqUKWnAgAF5/d57752WWGKJPGloixYt0qqrrlrl9e3bt88/qy8HAIByEUgHAABq1W677ZYmTJiQTjrppDzBaM+ePdOjjz5aMQHpmDFjUuPGbo4FAGDeIZAOAADUusMOOyw/ajJs2LCSr73hhhvqqFUAAPDzSAMBAAAAAIASBNIBAAAAAKAEgXQAAAAAAChBIB0AAAAAAEoQSAcAAAAAgBIE0gEAAAAAoKEH0ocPH56aNGmSttlmm3I3BQAAAACAeUyDCKRfe+216fDDD0///Oc/06efflru5gAAAAAAMA+Z7wPp33zzTbrjjjvSwQcfnDPSb7jhhirr77///rTccsulFi1apD59+qQbb7wxNWrUKE2cOLFim+eeey5ttNFGqWXLlqlLly7pD3/4Q5oyZUoZzgYAAAAAgLltvg+k33nnnWnFFVdMK6ywQtpzzz3TddddlwqFQl43atSotPPOO6cddtgh/ec//0kHHnhgOv7446u8/sMPP0xbbbVV+s1vfpNef/31HJSPwPphhx02y2NOnTo1TZ48ucoDAAAAAIB5U+OGUNYlAughAuKTJk1KzzzzTH5+1VVX5QD7Oeeck3/+9re/Tfvss0+V1w8dOjTtscce6cgjj8yZ6+uvv366+OKL00033ZS+//77Go8Zr2nXrl3FI7LYAQAAAACYN83XgfSRI0emF198Me2+++75edOmTdNuu+2Wg+vF9WuttVaV16y99tpVnkemepSDadOmTcWjb9++acaMGTmjvSaDBg3KAfvi4+OPP66zcwQAAAAAoG41TfOxCJj/+OOPafHFF69YFmVdmjdvni699NLZrrEeJV+iLnp1Sy65ZI2vif3HAwAAAACAed98G0iPAHqUXznvvPPSlltuWWVd1ES/7bbbcjmXhx9+uMq6l156qcrz1VdfPb399ttp2WWXnSvtBgAAAACgfplvA+kPPvhg+uqrr9J+++2X65RXFhOHRrZ6TER6/vnnp2OPPTZv99prr+UyLqFRo0b5Z6xbd9118+Si+++/f2rdunUOrD/xxBOzndUOAAAAAMC8a76tkR6B8s0333ymIHoxkP7yyy+nr7/+Ot19993pb3/7W+rRo0e64oor0vHHH5+3KZZmieUxOel7772XNtpoo9SrV6900kknVSkXAwAAAADA/Gu+zUh/4IEHZrkuJhSNWunFQPn2229fse70009Pv/rVr1KLFi0qlsWEpI8//ngdtxgAAAAAgPpovg2kz67LL788B8oXWWSR9Pzzz6dzzjknl3EBAAAAAIDQ4APp77//fjrttNPSl19+mZZccsl01FFHpUGDBpW7WQAAAAAA1BMNPpB+wQUX5AcAAAAAADSoyUYBAAAAAKA2CKQDAAAAAEAJAukAAAAAAFCCQDoAAAAAAJQgkA4AAAAAACUIpAMAAAAAQAkC6QAAAAAAUIJAOgAAAAAAlCCQDgAAAAAAJQikAwAAAABACQLpAAAAAABQgkA6AAAAAACUIJAOAAAAAAAlCKQDAAAAAEAJAukAAAAAAFCCQDoAAAAAAJQgkA4AAAAAACU0LbWS2vXmkL6pbdu25W4GAAAAAABzQEY6AAAAAACUIJAOAAAAAAAlCKQDAAAAAEAJAukAAAAAAFCCQDoAAAAAAJQgkA4AAAAAACUIpAMAAAAAQAkC6QAAAAAAUIJAOgAAAAAAlCCQDgAAAAAAJQikAwAAAABACQLpAAAAAABQgkA6AAAAAACU0LTUSmrXqoMfS42btyp3MwCoY6PP3KbcTQAAAABqkYx0AAAAAAAoQSAdAAAAAABKEEgHAAAAAIASBNIBAAAAAKAEgXQAAAAAAChBIB0AAAAAAEoQSAcAAAAAgBIE0gEAAAAAoASBdAAAAAAAKEEgHQAAAAAAShBIBwAAAACAEgTSAQAAAACgBIF0AAAAAAAoQSAdAAAAAABKEEgHAAAAAIASBNIBAAAAAKAEgXQAAAAAAChBIB0AAAAAAEoQSAcAAAAAgBIE0gEAAAAAoASBdAAAAAAAKEEgHQAAAAAAShBIBwAAAACAhhRI32effVKjRo1menzwwQflbhoAAAAAAPOgpmk+tNVWW6Xrr7++yrJFF110jvYxffr0HIBv3Hi++64BAAAAAIA5MF9GiZs3b546depU5XHRRRel7t27p9atW6cuXbqkQw45JH3zzTcVr7nhhhtS+/bt0/33359WXnnlvI8xY8akqVOnpqOPPjotscQS+bXrrLNOGjZsWMnjx2smT55c5QEAAAAAwLxpvgyk1yQyyy+++OL01ltvpRtvvDH94x//SMccc0yVbb799tt01llnpWuuuSZvt9hii6XDDjssDR8+PN1+++3p9ddfT7vsskvOeH///fdneayhQ4emdu3aVTwicA8AAAAAwLxpvizt8uCDD6Y2bdpUPN96663TXXfdVfG8W7du6bTTTksHHXRQuvzyyyuW//DDD/n5aqutlp9HRnqUiImfiy++eF4W2emPPvpoXn7GGWfUePxBgwalgQMHVjyPjHTBdAAAAACAedN8GUjv06dPuuKKKyqeR0mWJ598MmeKv/vuuzmw/eOPP6bvv/8+Z6G3atUqb9esWbPUo0ePite98cYbuVb68ssvP1PplkUWWWSWx4+yMPEAAAAAAGDeN18G0iNwvuyyy1Y8Hz16dNp2223TwQcfnE4//fS08MILp+eeey7tt99+adq0aRWB9JYtW+YJRouihnqTJk3SiBEj8s/KKme8AwAAAAAw/5ovA+nVRSB8xowZ6bzzzsu10sOdd975k6/r1atXzkgfP3582mijjeZCSwEAAAAAqG8axGSjkZ0e9c8vueSS9NFHH6Wbb745XXnllT/5uijpsscee6S99947/e1vf0ujRo1KL774Yi4R89BDD82VtgMAAAAAUF4NIpAek4eef/756ayzzkqrrrpquuWWW3IwfHbEpKIRSD/qqKPSCiuskHbYYYf00ksvpSWXXLLO2w0AAAAAQPk1KhQKhXI3Yn4Xk5u2a9cudTnyztS4+f+rxw7A/Gv0mduUuwnAfDaOnDRpUmrbtm25m1Nv6SeA+m/SkCHlbgJQz7UbPLhejyMbREY6AAAAAAD8XALpAAAAAABQgkA6AAAAAACUIJAOAAAAAAAlCKQDAAAAAEAJAukAAECtu+yyy1K3bt1SixYt0jrrrJNefPHFWW579dVXp4022igttNBC+bH55puX3B4AAOY2gXQAAKBW3XHHHWngwIFp8ODB6ZVXXkmrrbZa6tu3bxo/fnyN2w8bNiztvvvu6emnn07Dhw9PXbp0SVtuuWX65JNP5nrbAQCgJgLpAABArTr//PPTAQcckAYMGJBWXnnldOWVV6ZWrVql6667rsbtb7nllnTIIYeknj17phVXXDFdc801acaMGempp56a620HAICaCKQDAAC1Ztq0aWnEiBG5PEtR48aN8/PINp8d3377bfrhhx/SwgsvXOP6qVOnpsmTJ1d5AABAXRJIBwAAas3nn3+epk+fnjp27FhleTwfO3bsbO3j2GOPTYsvvniVYHxlQ4cOTe3atat4RCkYAACoSwLpAABAvXHmmWem22+/Pd177715otKaDBo0KE2aNKni8fHHH8/1dgIA0LA0LXcDAACA+UeHDh1SkyZN0rhx46osj+edOnUq+dpzzz03B9KffPLJ1KNHj1lu17x58/wAAIC5RUY6AABQa5o1a5bWWGONKhOFFicOXW+9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AAAAAABKEEgHAAAAAIASBNIBAIBad9lll6Vu3bqlFi1apHXWWSe9+OKLJbe/66670oorrpi37969e3r44YfnWlsBAOCnCKQDAAC16o477kgDBw5MgwcPTq+88kpabbXVUt++fdP48eNr3P6FF15Iu+++e9pvv/3Sq6++mnbYYYf8ePPNN+d62wEAoCaNCoVCocY11JpJkyal9u3bp48//ji1bdu23M0BAGAeMXny5NSlS5c0ceLE1K5duzSviAz0tdZaK1166aX5+YwZM/J5HH744em4446bafvddtstTZkyJT344IMVy9Zdd93Us2fPdOWVV860/dSpU/Oj8nh7ySWXLMt4+/73x87V4wHznu2X61TuJtQLk4YOLXcTgHqu3aBB9Xq83XSutaoB++KLL/LPeFMAAGBOff311/NMIH3atGlpxIgRaVClC6HGjRunzTffPA0fPrzG18TyyGCvLDLY77vvvhq3Hzp0aBoyZMhMy423AQDmYWeeWa/H2wLpc8HCCy+cf44ZM2aeuQCa1xS/PZL1Xzf0b93Tx3VPH9ct/Vv39HHD7OO4eTQG9YsvvniaV3z++edp+vTpqWPHjlWWx/N33323xteMHTu2xu1jeU0iSF858B4Z719++WVaZJFFUqNGjWrlPGB++TsCUB/5e0l9MSfjbYH0uSAycEIE0f1xqFvRv/q47ujfuqeP654+rlv6t+7p44bXxxIxZta8efP8qCxKKUJ9Ud/+jgDUV/5eMi+Nt002CgAA1JoOHTqkJk2apHHjxlVZHs87daq5TnAsn5PtAQBgbhNIBwAAak2zZs3SGmuskZ566qkqpVfi+XrrrVfja2J55e3DE088McvtAQBgblPaZS6I204HDx480+2n1B59XLf0b93Tx3VPH9ct/Vv39HHd08e1J+qX9+/fP6255ppp7bXXThdeeGGaMmVKGjBgQF6/9957pyWWWCJPGhqOOOKI1Lt373TeeeelbbbZJt1+++3p5ZdfTn/5y1/KfCYwZ/wdAZg9/l4yL2pUiIrqAAAAtejSSy9N55xzTp4wtGfPnuniiy9O66yzTl63ySabpG7duqUbbrihYvu77rornXDCCWn06NFpueWWS2effXbq169fGc8AAAD+j0A6AAAAAACUoEY6AAAAAACUIJAOAAAAAAAlCKQDAAAAAEAJAukAAABQR2IC3UaNGqXXXnut3E0BmC/EhOUXXnhhuZtBAySQ/jNddtll+R9uixYt0jrrrJNefPHFktvfddddacUVV8zbd+/ePT388MNV1secryeddFLq3LlzatmyZdp8883T+++/nxqq2u7fffbZJw9eKz+22mqr1JDNSR+/9dZb6Te/+U3ePvpuVv/DmtP3bX5X23188sknz/Q5js99QzUn/Xv11VenjTbaKC200EL5EX9jq2/v73Dd97G/xb+sj//2t7+lNddcM7Vv3z61bt069ezZM918881VtvE5rtv+9RmGhqP47/2ggw6aad2hhx6a18U2APO7msY/8fjggw/K3TSY6wTSf4Y77rgjDRw4MA0ePDi98sorabXVVkt9+/ZN48ePr3H7F154Ie2+++5pv/32S6+++mraYYcd8uPNN9+s2Obss89OF198cbryyivTv//973wBF/v8/vvvU0NTF/0b4kL3s88+q3jcdtttqaGa0z7+9ttv09JLL53OPPPM1KlTp1rZ5/yuLvo4rLLKKlU+x88991xqiOa0f4cNG5b/Tjz99NNp+PDhqUuXLmnLLbdMn3zyScU2/g7XfR8Hf4t/fh8vvPDC6fjjj8/9+/rrr6cBAwbkx2OPPVaxjc9x3fZv8BmGhiP+X3b77ben7777rmJZ/D299dZb05JLLlnWtgHMTdXHP/FYaqmlyt0smPsKzLG11167cOihh1Y8nz59emHxxRcvDB06tMbtd91118I222xTZdk666xTOPDAA/PvM2bMKHTq1KlwzjnnVKyfOHFioXnz5oXbbrut0NDUdv+G/v37F37961/XYavn7z6urGvXroULLrigVvc5P6qLPh48eHBhtdVWq/W2zot+6eftxx9/LCy44IKFG2+8MT/3d7ju+zj4W1z7fzd79epVOOGEE/LvPsd127/BZxgajuK/91VXXbXw17/+tWL5LbfcUujRo0deF9uERx55pLDBBhsU2rVrV1h44YXztckHH3xQ8ZpRo0YV4tL71VdfrVj2xhtvFLbaaqtC69atC4sttlhhzz33LEyYMGEunyXATys1/rnvvvvyeCnGm0sttVTh5JNPLvzwww8V6+Nv35VXXpn/LrZs2bKw4oorFl544YXC+++/X+jdu3ehVatWhfXWW6/K38z4ffvtt89/G+Nv5Jprrll44oknSl4zf/XVV4X99tuv0KFDh3wN0qdPn8Jrr71WJ/1BwyYjfQ5NmzYtjRgxIt8qXdS4ceP8PDKYahLLK28fIiOquP2oUaPS2LFjq2zTrl27fAvyrPY5v6qL/q2cLbnYYoulFVZYIR188MHpiy++SA3Rz+njcuxzXlaX/RElGhZffPGcvb7HHnukMWPGpIamNvo37gD44YcfcgZq8He47vu4yN/i2unjuC556qmn0siRI9PGG2+cl/kc123/FvkMQ8Oy7777puuvv77i+XXXXZfvVqlsypQp+Q6Yl19+Of/tiL83O+64Y5oxY0aN+5w4cWLadNNNU69evfJrHn300TRu3Li066671vn5ANSWZ599Nu29997piCOOSG+//Xa66qqr0g033JBOP/30KtudeuqpebuYJyJKk/7ud79LBx54YBo0aFD+GxjjrsMOO6xi+2+++Sb169cv/z2NqgORDb/ddtuVvPbdZZdd8l2HjzzySB4Drr766mmzzTZLX375ZZ32AQ1P03I3YF7z+eefp+nTp6eOHTtWWR7P33333RpfExe1NW0fy4vri8tmtU1DURf9G+IP70477ZRvPfrwww/Tn//857T11lvni+kmTZqkhuTn9HE59jkvq6v+iGBYDEwieBO30g0ZMiTXpI4yRgsuuGBqKGqjf4899tj8hUQxyObvcN33cfC3+Jf38aRJk9ISSyyRpk6dmvvs8ssvT1tssUVe53Nct/0bfIah4dlzzz1zsOe///1vfv7888/nci/xpVpRzHNTWQTbF1100RxYWnXVVWfa56WXXpqD6GeccUaV10Qpmffeey8tv/zydXpOAHPqwQcfTG3atKl4HuOfr776Kh133HGpf//+eVkke0XQ/Jhjjsml9Yriy8fiF4VxjbDeeuulE088MSdAhgjEV/6CMsrxxaMo9nnvvfem+++/v0rAvSjKncY8OBFIb968eV527rnnpvvuuy/dfffd6fe//32d9AkNk0A6DcJvf/vbit9jMtIePXqkZZZZJg+A41tKmBfEYKUoPsMRWO/atWu688478xwBzJ6oQ1+8AI4JCJl7fexv8S8XX5pFNk9k6kSWTmRAxkXLJptsUu6mNYj+9RmGhicC4ttss01OZoisyfi9Q4cOVbaJOwZjoueYmyK+yCtmokf2ZE2B9P/85z95TpHKQami+JJOIB2ob/r06ZOuuOKKiucxD0+Mg+LLxcoZ6JHIEHNJxN2prVq1ystiu6JikkOMoyovi9dMnjw5tW3bNo/DTj755PTQQw/lBLIff/wxz1Uxq4z0+Jsar1lkkUWqLI/XxN9UqE0C6XMoBk2RcRS33lUWz2c1QWAsL7V98Wcs69y5c5VtevbsmRqSuujfmsRFcRwrZpluaBe+P6ePy7HPednc6o/27dvnC62GNlv6L+nfyEyIIO+TTz5ZZUDn73Dd93FN/C2e8z6OcgHLLrts/j0+m++8804aOnRoDvT6HNdt/9akIX+GoaGVdylmQV522WUzrY+SA5HccPXVV+e7sSKQHgH0KDNVkwj4xGvOOuusmdZV/vsNUF9E4Lw4Rqr8tyzuko679aqrnEyzwAILVPzeqFGjWS4rfgl59NFHpyeeeCJfV8QxW7ZsmXbeeeeSf1Pjb2flO4UqXzNDbVIjfQ41a9YsrbHGGjlLqSj+scfzuD2lJrG88vYh/igUt4/bg+OirvI28U1cZDTMap/zq7ro35r873//yzVNG+JA9ef0cTn2OS+bW/0RA4b4hr2hfY5/bv+effbZ+bbAqEO65pprVlnn73Dd93FN/C3+5X8n4jVRhiT4HNdt/9akIX+GoSGJsk4RwIm5P4qlCIrib0DMp3DCCSfkL9RWWmmlXO6glKjd+9Zbb6Vu3brlIFHlRwSrAOYF8bcs/v5V/zsWj0hO+Lkiy32fffbJc01E5nqMb0ePHl2yHVHGsGnTpjO1o/odRPCLlXu203nR7bffnmckvuGGGwpvv/124fe//32hffv2hbFjx+b1e+21V+G4446r2P75558vNG3atHDuuecW3nnnncLgwYMLCyywQJ6pvejMM8/M+/j73/9eeP311/OMyDHj8XfffVdoaGq7f7/++uvC0UcfXRg+fHhh1KhRhSeffLKw+uqrF5ZbbrnC999/X2iI5rSPp06dWnj11Vfzo3Pnzrk/4/eYaXt299nQ1EUfH3XUUYVhw4blz3F87jfffPM8K/n48eMLDc2c9m/8jW3WrFnh7rvvLnz22WcVj/j7UHkbf4frro/9Lf7lfXzGGWcUHn/88cKHH36Yt4//78X//66++uqKbXyO665/fYahYenfv3/+G1o0adKk/CiKdbHN9OnTC4ssskhhzz33zOO2p556qrDWWmsV4lL73nvvzdvG34x4HmO78MknnxQWXXTRws4771x48cUXCx988EHh0UcfLeyzzz6FH3/8sQxnCzD7fw+L4u9WjJVOPvnkwptvvpnHT7fddlvh+OOPr9im8t/Cmv4ehqeffjov++qrr/LzHXfcsdCzZ8+8zWuvvVbYbrvtCgsuuGDhiCOOqHhN165dCxdccEH+fcaMGYUNN9ywsNpqqxUee+yxiuvlP//5z4WXXnqpzvqFhkkg/We65JJLCksuuWQOGqy99tqFf/3rXxXrevfunf/QVHbnnXcWll9++bz9KqusUnjooYeqrI9/+CeeeGKhY8eO+aJvs802K4wcObLQUNVm/3777beFLbfcMg9WI8Aef3APOOCABhvg/Tl9XPyfXfVHbDe7+2yIaruPd9tttxxkj/0tscQS+XlceDVUc9K/8e++pv6NL96K/B2u2z72t/iX93FclCy77LKFFi1aFBZaaKHCeuutl4PFlfkc113/+gxDwzKrwFH1QHp44oknCiuttFL+u9ujR4+c+FAqkB7ee++9HCyKL/hatmxZWHHFFQtHHnlk/jsOMK/8PYxg+vrrr5//jrVt2zaPt/7yl7/8okB6bNOnT5+8zy5duhQuvfTSPG6bVSA9TJ48uXD44YcXFl988TxOi9ftsccehTFjxtR6f9CwNYr//PK8dgAAAAAAmD+pkQ4AAAAAACUIpAMAAAAAQAkC6QAAAAAAUIJAOgAAAAAAlCCQDgAAAAAAJQikAwAAAABACQLpAAAAAABQgkA6AAAAAACUIJAOAAAAAAAlCKQDNCD77LNP2mGHHVJ9NHr06NSoUaP02muvlbspAAAAAFUIpANQdtOmTSt3EwAAAABmSSAdoIHaZJNN0uGHH56OPPLItNBCC6WOHTumq6++Ok2ZMiUNGDAgLbjggmnZZZdNjzzySMVrhg0blrPGH3roodSjR4/UokWLtO6666Y333yzyr7vueeetMoqq6TmzZunbt26pfPOO6/K+lh26qmnpr333ju1bds2/f73v09LLbVUXterV698jGhfeOmll9IWW2yROnTokNq1a5d69+6dXnnllSr7i+2vueaatOOOO6ZWrVql5ZZbLt1///1VtnnrrbfStttum48X57bRRhulDz/8sGJ9vH6llVbK57Tiiiumyy+/vBZ7GwAAAJiXCaQDNGA33nhjDlC/+OKLOah+8MEHp1122SWtv/76OVi95ZZbpr322it9++23VV73pz/9KQfHI8i96KKLpu222y798MMPed2IESPSrrvumn7729+mN954I5188snpxBNPTDfccEOVfZx77rlptdVWS6+++mpeH20ITz75ZPrss8/S3/72t/z866+/Tv3790/PPfdc+te//pWD5P369cvLKxsyZEg+7uuvv57X77HHHunLL7/M6z755JO08cYb58D+P/7xj9zGfffdN/344495/S233JJOOumkdPrpp6d33nknnXHGGblN0T8AAAAAjQqFQqHcjQBg7tVInzhxYrrvvvtyxvf06dPTs88+m9fF75HxvdNOO6WbbropLxs7dmzq3LlzGj58eM48j4z0Pn36pNtvvz3ttttueZsIVv/qV7/KgfIIZEcAe8KECenxxx+vOO4xxxyTs9gjK7yYkR6Z5/fee2+VGumRlR6B9Z49e87yHGbMmJHat2+fbr311pxhXsxIP+GEE3KWe4is+jZt2uRs+q222ir9+c9/zm0eOXJkWmCBBWbaZ2Tex2t33333imWnnXZaevjhh9MLL7zwi/sdAAAAmLfJSAdowKI8S1GTJk3SIosskrp3716xLMq9hPHjx1d53XrrrVfx+8ILL5xWWGGFnMkd4ucGG2xQZft4/v777+dgfdGaa645W20cN25cOuCAA3ImegT6ozTLN998k8aMGTPLc2ndunXertjumMA0SrnUFESPoHuUeNlvv/1y8L34iEB65dIvAAAAQMPVtNwNAKB8qgeWI7O78rJ4XswCr20R7J4dUdbliy++SBdddFHq2rVrLs8SgfzqE5TWdC7Fdrds2XKW+4+gfIj68Ouss06VdfHlAgAAAIBAOgBzLGqVL7nkkvn3r776Kr333nt5os4QP59//vkq28fz5ZdfvmRgulmzZvln5az14mtj4s+oex4+/vjj9Pnnn89ReyNbPeqdRx336gH3yLpffPHF00cffZTL0gAAAABUJ5AOwBw75ZRTchmYCEIff/zxecLSHXbYIa876qij0lprrZVrjkcd9aivfumll+ZgeCmLLbZYzhx/9NFHc831Fi1a5FIuUdLl5ptvzqVgJk+enCc6LZVhXpPDDjssXXLJJXkC1EGDBuX9xpcBa6+9di5LExOV/uEPf8jLo6b61KlT08svv5y/JBg4cOAv6isAAABg3qdGOgBz7Mwzz0xHHHFEWmONNfKEpA888EBFRvnqq6+e7rzzzjy556qrrppOOumkHHiPiU5Ladq0abr44ovTVVddlTPEf/3rX+fl1157bQ5ox3732muvHPCOoPuciKD/P/7xj1zGpXfv3rndUcqlmJ2+//77p2uuuSZdf/31uUZ8bBOTp8bkpwAAAACNCoVCodyNAGDeMGzYsNSnT58c2G7fvn25mwMAAAAwV8hIBwAAAACAEgTSAQAAAACgBKVdAAAAAACgBBnpAAAAAABQgkA6AAAAAACUIJAOAAAAAAAlCKQDAAAAAEAJAukAAAAAAFCCQDoAAAAAAJQgkA4AAAAAACUIpAMAAAAAQAkC6QAAAAAAUIJAOgAAAAAAlCCQDgAAAAAAJQikAwAAAABACQLpAAAAAABQgkA60GDts88+qVu3bnV6jEaNGqWTTz45zU+iz7bddttyN4MymR8/0wDA/G306NF5DHPDDTfU+bHiGHGsOGY5xs/Dhg3Lx4+fc1tDvk6o6X2vbZtsskl+AOUjkA7MFW+88UbaeeedU9euXVOLFi3SEksskbbYYot0ySWXpIauOLAvPho3bpwWXnjhtPXWW6fhw4f/7P1efvnlc+ViYW70SZMmTdKSSy6Zdtxxx/Taa6+Vu3nzpejXPffcM3Xp0iU1b948fwY333zzdP3116fp06eXu3kAQJnF2DLGZeuss065m1JlnNi0adM8blljjTXSEUcckd5+++1aO059Hk/X57bVtQkTJuT3esUVV0wtW7ZMiy22WFp77bXTsccem7755ptyNw+YjzUqFAqFcjcCmL+98MILqU+fPjkQ2r9//9SpU6f08ccfp3/961/pww8/TB988EFZ2vXDDz+kGTNm5KBhXYnB/eDBg0tm8EbQeKmllkq777576tevXw5avvfee3lw/N1336WXXnopde/efY6Pveqqq6YOHTrUejZKZJrEvh988MFUV2rqk3feeSddccUVaerUqfmz07Nnzzo7fkNzzTXXpIMOOih17Ngx7bXXXmm55ZZLX3/9dXrqqafSQw89lE477bT05z//ebY/0wDA/GeDDTZIn376aR6nvf/++2nZZZctW1tiPBJJOXvvvXeKkMakSZPSf/7zn3TXXXelKVOmpLPOOisNHDiwYvvYJsaQCyywQE7QqMvxdIxb4zojrjGinXU1fp5V2+L6Ztq0aalZs2Y5QWdumhvXCV9++WXq1atXmjx5ctp3331zMP2LL75Ir7/+ej5u/Kzru45n932vbcVs9HLcbQD8P03//58Adeb0009P7dq1ywHh9u3bV1k3fvz4WjtODJpbt24929vHQLo+WX311XNGcNFGG22Us9IjeBxB9Yaoep/EBdz222+f++Sqq64qa9vmJd9++21q1apVjeviS4kIoq+33nrp4YcfTgsuuGDFuiOPPDK9/PLL6c0335yLrQUA6ptRo0bl5Ji//e1v6cADD0y33HJL/mK9nJZffvkq48Rw5plnpu222y4dddRROcAaCRkhAptxV2xdKl6LRKB+ToL1tS2C53V9ruV07bXXpjFjxqTnn38+rb/++lXWRXA9vkCoDd9///0cfRlR7vcdmDuUdgHqXGSdr7LKKjMF0UPchjc7tQur12WO32NZ3Lr5u9/9Li200EJpww03TOeee25e/t///nemfQwaNCgPhr766quZaqRH9kDcEjpgwICZXhcDshiMHn300fl5ZHicdNJJ+fbR+IIgBswR9H766adTbYp9Fvuvsii1semmm+a+i4yHlVdeOQeWK4vzeuutt9IzzzxTcdtr5Xp6EydOzEHSYhmPyCiKzJ3IYJldjz/+eM4Kj76JNsSFVdFHH32Uj3nBBRfM9Lq4CIt1t912W5pTcd7Fi7nw97//PW2zzTZp8cUXz+exzDLLpFNPPXWmUiSRNfWb3/wm3w0R7f3Vr36Vfvvb3+bspaInnngif4bic9qmTZu0wgorVGRhF0UmU1w0Rn/F8aL/jjnmmLy8sji/ww47LN133305Kye2jX8Djz766EznFBkla665Zm5XtD++ICh+vqv761//mj93cQtrfF7jHOLujsrifY5jjhgxIm288cY5gF79PCobMmRIPlZcEFcOohdF2+LfyqzEv7VDDjkk91e0a5FFFkm77LLLTPUh499YHCuy3eNcY7vo7+j3orFjx+Z/g/H+RJ917tw5/frXv67TWpMAwE+LcUKMt2PcFeUa43lNIjM47m5r27ZtHlPF3aiRKV7TGP/dd9/N+4oxTYwNYsxx//33/6J2xvji9ttvz+VeIpmn1HXGT407So2ni/WwY12Mg2JcHvupvK6m8Uup8XOY1Riw+j5LtW1WNdIjW784joxM9vgS4pNPPqmyTYz5Yhwcy3fYYYf8+6KLLpqvg+ak1F9dXifEtVEErNddd92Z1sXnrvKXCNFPNY1jq9caL/ZZfHZOOOGEXIY0xtCvvPJKXn7jjTfOtI/HHnssrytm31d/j6JW/NJLL13jOUQCS3ze5+T6DqgfZKQDdS7qoket78hqjQBfbYqAXQTmzjjjjHzLZgxYIrB55513pj/96U9Vto1lW265Zb4IqCk7PepvxyAvApmVMxkiGBqB0ghaFgPrUQojyo4ccMABuQRGZEb07ds3vfjii7VWcqQ4CKve3hhURVA2MrPjIuGBBx7IA/gIgh966KF5mwsvvDAdfvjhefB7/PHH52VRtqOYndy7d+88QI6Moii5E4PW+KLhs88+y6/9KRGY3m233XImc1wgxeAv3osIFMdttjFojOzxuMj64x//WOW1xYBtXKjMqeKXCnGRVBywxjnGrbvx8x//+Ef+kiPeo3POOafii494b+I9jD6JYHqcewx64wuF+DIkLkTis9OjR490yimn5AFslByKTJei6N/o8+eeey79/ve/TyuttFKu/R8XAVGKJz4nlcV28XmK9ybO9+KLL87B/MigKbb/1VdfTVtttVW+cIsgc1ygxPHjgqW6uBg88cQT06677pr233//XBsy5hiIYHnsp/IXVXERG3czxGc2LpKK73118VmI8i2xj/gc/Bxxp0l8fuJYcQEZn9v4jMbFSXzRVcyEjwvDoUOH5rZHDct4jyLbPS5Q4jMTon/ivYj3KS584o6VCLRHn5XjFl0A4P/GbzvttFMeI8cYOP5fH2OAtdZaq8pYKbLBYzx88MEH54zwSHqIsWJ18f/7GCtGwPK4447LiSkxVo/g7T333JPH5T9XjGlirBtJLjHeiOBqTX5q3FFqPF0U47wYt8X4MzLSf8n4eU7MTtsqizFzfGkQ71eMx8aNG5cuuuiiPNatPo6M8WiMnaMWfiQpPfnkk+m8887LCR/xvpb7OiGuLaONN998c42frV8iEnLiMx5fHMS1QwS0o73x2ax+rDvuuCNfp0Vf1ST6IEoPVf93EkkocUdo8Vpldq/vgHoiaqQD1KXHH3+80KRJk/xYb731Csccc0zhscceK0ybNq3KdqNGjYo5GwrXX3/9TPuI5YMHD654Hr/Hst13332mbeMYa6yxRpVlL774Yt7+pptuqljWv3//QteuXSueR5timwceeKDKa/v161dYeumlK57/+OOPhalTp1bZ5quvvip07NixsO+++5Zsd02K5z1k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BCwICERA2O96z//bdReWw4Luzs7ufV/XgZ0zZ2bePTOze+a3z3neDiWHHHJIyejRoxfa/rrrrsteW2ks22+/fcmbb76Z7avyn41K91neOD/88MOy19xLL7200PWVPUel0r5O13Xv3r3S+67q57sFxw3UvAbpn0KH+QDUbWnCplSFPGLEiEIPpSikiq53330369UJAFDMUhu7VD380ksvZWfhQXk+JwDFRI90AKpVauGR2pSkUzepvP9oeSk8T/MHpDYoAADFfFyT2tallolpLqTNN9+8YOOidvI5ASg2KtIBqBb/+te/sokqr7vuuvjyyy/j448/znoKUlGarCj1qk+9QT/55JOs12WaCOqtt97KepUDABSL1JM6henbbrttdjyT5hVKcyNdeeWV2VwvkPicABQrk40CUC3SxD5pEs9111037r77bgfHi5AmP0r7Z9KkSdGsWbPsg2f6sClEBwCKTZrsMoWjjz76aHz77bfZJJipIr10cnFIfE4AilW9q0h/4YUX4tprr83++plmQE4zc6detHmee+656NOnT9avtlOnTnHRRRdl1YMAAAAAANR99a5H+syZM2PTTTeNm2++uUrbjxs3Lvbee+/YZZddst5dZ555Zna62hNPPFHtYwUAAAAAoPDqXUV6eQ0aNFhsRfr5558fjz32WNbDq9Shhx4aU6dOjeHDh9fQSAEAAAAAKBQ90hdj5MiR0b179wrrevTokVWmL0qaVCUtpebPnx9TpkyJFVdcMQvvAQCgKlLNyzfffBMdO3aMhg3r3cmkVZaOtz/77LNYbrnlHG8DAFAtx9uC9MVIk7+1a9euwrp0efr06dls5C1atFjoNv37949LL720BkcJAEBdNmHChPjJT35S6GHUWilET3MZAQBAdR1vC9KrwYUXXphNTlpq2rRpseqqq2ZPyPLLL1/QsQEAUDxS8UYKiFOlNYtWun8cbwMAUF3H24L0xWjfvn1Mnjy5wrp0OR2gV1aNnjRr1ixbFpRu48AeAIAlpV1J1faP420AAKrreFujxcXYdtttY8SIERXWPfXUU9l6AAAAAADqvnoXpM+YMSPefvvtbEnGjRuXfT1+/PiytixHH3102fYnn3xyfPzxx3HeeefF+++/H3/4wx/ivvvui7POOqtg3wMAAAAAADWn3gXpb775Zmy22WbZkqRe5unrvn37ZpcnTpxYFqonq6++ejz22GNZFfqmm24a1113Xdx2223Ro0ePgn0PAAAAAADUnAYlJSUlNfh49bZpfatWrbJJR/VsBACgqhxHVo39BAD117x582Lu3LmFHga1VJMmTaJRo0Y/ynGkyUYBAAAAgKKSaoMnTZoUU6dOLfRQqOVat24d7du3r9KEonkE6QAAAABAUSkN0du2bRstW7b8wSEpdfOPLbNmzYrPP/88u9yhQ4cfdH+CdAAAAACgqNq5lIboK664YqGHQy3WokWL7P8UpqfXS16bl8Wpd5ONAgAAAADFq7QneqpEh8UpfZ380F76gnQAAAAAoOho50JNvk4E6QAAAAAAkEOQDgAAAADAj+6SSy6JLl26VPvjdO7cOW644YZqfQyTjQIAAAAAdUKvIW/U6OMNOnarJdr+2GOPjTvuuCP69+8fF1xwQdn6hx56KA444IAoKSlZovD4zDPPzJY8//jHP+Liiy+OV199NaZPnx7t27ePrl27xk033ZRNwFmdzjnnnOjdu3fUBSrSAQAAAABqSPPmzePqq6+Or7/+utof64svvojddtst2rRpE0888US89957cfvtt0fHjh1j5syZS32/c+bMqdJ2yy67bKy44opRFwjSAQAAAABqSPfu3bOq8FSVnuevf/1rbLjhhtGsWbOs+vy6664ru65bt27xySefxFlnnZVNprmoCTVffvnlmDZtWtx2222x2Wabxeqrrx677LJLXH/99dnXyZAhQ6J169YVbpcq5Mvf5yX/v0VLup90u/THgD/96U9ZID9//vwKt91vv/3i+OOPr3C75Mknn8xuN3Xq1Arbn3HGGbHrrruWXX7ppZdixx13jBYtWkSnTp3i9NNPrxD6f/7557Hvvvtm16exDB06NGqCIB0AAAAAoIY0atQorrzyyqy1yqefflrpNqNGjYqf//znceihh8Y///nPLJBO7VlS6J088MAD8ZOf/CQuu+yymDhxYrZUJgX23333XTz44INL1DamMmPHjs3C/fTYb7/9dhxyyCHx1VdfxbPPPlu2zZQpU2L48OFxxBFHLHT7VBmfAvt0H6XmzZsX9957b9n2H330Ueyxxx5x0EEHxTvvvJNdl4L10047rUJ7nAkTJmSP+5e//CX+8Ic/ZOF6dROkAwAAAADUoNQPPVVq9+vXr9LrBwwYkAXPKTxfZ511svA4hcnXXnttdn1q1ZIC+eWWWy4Ly9NSmW222SZ+9atfxeGHHx4rrbRS7Lnnntl9TJ48eanaudx5551ZZfsmm2wSK6ywQnZ/w4YNK9smBdvpcVLV+4LSeNMfBspvP2LEiKxCPQXnSarST6F66vu+9tprx3bbbRc33nhj9rjffvttfPDBB/H3v/89br311ux722KLLWLQoEHxv//9L6qbIB0AAAAAoIalPulp4tHUt3xBad32229fYV26/OGHH2ZV3EviN7/5TUyaNCkGDhyYtYpJ/6+33npZpfuSWG211WLllVeusC6F3qnCfPbs2dnl1GYlheUNG1YeO6ftn3vuufjss8/Ktt97773LWsukiVFT1X3qrV669OjRI2sfM27cuGy/NG7cOAvQS6XvZcHWNNVBkA4AAAAAUMN22mmnLCS+8MILq/2x0oSfqRXLb3/72yyMTr3N09dJCr0XbPsyd+7che5jmWWWWWhd6lWebvvYY49l7VZefPHFStu6lNpqq61izTXXjHvuuSerIk8tZ8pvP2PGjPjFL36RtY4pXVK4nv6AkG5XSI0L+ugAAAAAAPXUVVddlbV4WXfddSusX3/99bOJQstLl1Obl9QiJWnatOkSV6eX3i6F0qUTeKYq82+++Sa7XBqWpwC7Kpo3bx4HHnhgVlmeeqin72PzzTfPvU0KztP2qcd7CvFTRXqpdNt///vfsdZaa1V621R9nnq+px7yKZRPxowZs9AEptVBRToAAAAAQAFsvPHGWbCc+oCXd/bZZ2f9wy+//PKsL3hqAfP73/8+zjnnnLJtOnfuHC+88EL897//jS+//LLS+3/00UfjyCOPzP5P95NC51SJ/vjjj8d+++2XbdO1a9do2bJl1ks9TfaZepiXTmpaFUcccURWkT548ODcavTy248ePTprOXPwwQdHs2bNyq47//zz45VXXsn6wacwP1WiP/zww2WTjaagPk1GmqrWX3vttSxQP+GEE6JFixZR3VSkAwAAAAB1wqBj/69KuZhcdtllce+991ZYlyqz77vvvujbt28Wpnfo0CHbLk06Wv52KVBO1eWpR/mC7VmSDTbYIAvJUzCfWq+k0DpN4nnbbbfFUUcdVTZx6V133RXnnntuNolnmuT0kksuiZNOOqlK4991112z+0ghfZrUdHFStfnWW28dr7/+etxwww0VrkuTmD7//PPx61//Onbcccfse0rfX8+ePcu2uf3227PwfOedd4527drFFVdckU3KWt0alFS2h/lRTZ8+PVq1ahXTpk2L5ZdfvtDDAQCgSDiOrBr7CWqPXkPeqLHHKsawDPhxfPvtt9nEk6uvvnrWWgSW9vWyJMeRWrsAAAAAAEAOQToAAAAAAOQQpAMAAAAAQA5BOgAAAAAA5BCkAwAAAABADkE6AAAAAADkEKQDAAAAAEAOQToAALDUXnjhhdh3332jY8eO0aBBg3jooYcqXJ/WVbZce+21Zdt07tx5oeuvuuqqAnw3AABQOUE6AACw1GbOnBmbbrpp3HzzzZVeP3HixArL4MGDs6D8oIMOqrDdZZddVmG73r1719B3AAAAi9e4CtsAAABUas8998yWRWnfvn2Fyw8//HDssssuscYaa1RYv9xyyy207aLMnj07W0pNnz59iccNAFDfPPfcc9lx2Ndffx2tW7eutsc59thjY+rUqQudqVjsBOkAAECNmDx5cjz22GNxxx13LHRdauVy+eWXx6qrrhqHH354nHXWWdG4ceUfV/r37x+XXnppDYwYACg6w3rW7OMdfu8S3+SLL76Ivn37ZsdF6fhohRVWyM7wS+u23377qC7bbbddduZfq1atqu0x6jJBOgAAUCNSgJ4qzw888MAK608//fTYfPPNo02bNvHKK6/EhRdemH3IGzBgQKX3k67v06dPhYr0Tp06Vfv4AQB+DKnF3Zw5c7Jjo3SWXgrTR4wYEV999dVS3V9JSUnMmzdvkUUIpZo2bVrlMwBZmB7pAABAjUj90Y844oho3rx5hfUpFO/WrVtssskmcfLJJ8d1110XN910U4X2LeU1a9Ysll9++QoLAEAxSC1PXnzxxbj66quzNiurrbZabL311lmhwM9+9rP4z3/+k80n8/bbb1e4TVqXWrMk6f90+e9//3tsscUW2bFR6Tw077//foXHu/7662PNNdescLt0f6kQoUWLFtl9lPfggw9mhQ+zZs3KLk+YMCF+/vOfZ61gUtHDfvvtl42xVArw07Fcun7FFVeM8847Lwv26yJBOgAAUO3SB8YxY8bECSecsNhtu3btGt99912FD2kAAHXBsssumy2pf/iiigaq6oILLsja47333ntx8MEHx5ZbbhlDhw6tsE26nNrmLSgVIuyzzz4xbNiwhbbff//9o2XLljF37tzo0aNH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pUqYF198MTbYYIP43//+l/VEf+CBB2K//faLYjJ9+vRo1apVTJs2LauyqUlXvfVljT4eLK0LNjNxMADU5HHknDlz4qijjsomF23cuHG2bv78+XH00UdnoXfTpk2X+D5TRfqDDz4Y+++/f3Y5fdRIlepnn312nHPOOdm69L20a9cuhgwZkgX27733Xna8n4pmttxyy2yb4cOHZwU1n376aXb7Bc2ePTtbyu+nTp06FeR4G6io15A3auyxBh27VY09FgB105Icb9fqivSvv/66bAKkVHmeJhndaKONCj0sAAAoeikov/fee+P999+PoUOHZgUrH330UQwePHipQvTKjBs3LiZNmpS1cymVPqh07do1Ro4cmV1O/6d2LqUhepK2b9iwYVbBvqiJUtP9lC4pRAcAgOr0f6UntVhq4ZIOvksrWsaMGZOdPlreJptsUqDRAQBAcVtnnXWypTqUHsenCvTy0uXS69L/qV97ealCvk2bNmXbLOjCCy/MWj8uWJEOAAD1NkjfbbfdsgC91D777FN22mhan/5P/RwBAICqS8fQqb3KiBEj4vPPP8/aupSX+qXXVs2aNcsWAACoKbU6SE+nggIAAD++NKloCtL33nvvrH1iKlD5sbVv3z77f/LkydGhQ4ey9elyly5dyrZJQX553333XUyZMqXs9gAAUGi1OkhfbbXVCj0EAACok+6555647777skk9q8vqq6+eheGp6r00OE9tWFLv81NOOSW7vO2228bUqVNj1KhRscUWW5RVw6cK+dRLHQAAaoNaHaQDAADVI00outZaa/3g+5kxY0aMHTu2wlmlb7/9dtbjfNVVV40zzzwzrrjiilh77bWzYP3iiy+Ojh07xv77759tv/7668cee+wRJ554YgwcODDmzp0bp512Whx66KHZdgAAUBs0LPQAAACAmnf22WfH7373uwrzES2NN998MzbbbLNsSdIkoOnrvn37ZpfPO++86N27d5x00kmx1VZbZcH78OHDo3nz5mX3MXTo0FhvvfWy+ZFShfwOO+wQf/rTn37gdwgAAD8eFekAAFAPvfTSS/Hss8/G3//+99hwww2jSZMmFa5/4IEHqnQ/3bp1yw3jU+/1yy67LFsWJVWvDxs2bAlGDwAANUuQDgAA9VDr1q3jgAMOKPQwAACgKAjSAQCgHrr99tsLPQQAACgatTZIT30V02mgVTF69OhqHw8AANQ13333XTz33HPx0UcfxeGHHx7LLbdcfPbZZ7H88svHsssuW+jhAQBArVFrg/T999+/0EMAAIA665NPPok99tgjxo8fH7Nnz46f/vSnWZB+9dVXZ5cHDhxY6CECAECtUWuD9H79+hV6CAAAUGedccYZseWWW8Y//vGPWHHFFcvWp77pJ554YkHHBgAAtU2tDdIBAIDq8+KLL8Yrr7wSTZs2rbC+c+fO8d///rdg4wIAgNqoKIL0efPmxfXXXx/33XdfdurpnDlzKlw/ZcqUgo0NAACK0fz587Pj7AV9+umnWYsXAADgew2jCFx66aUxYMCA6NmzZ0ybNi369OkTBx54YDRs2DAuueSSQg8PAACKzu677x433HBD2eUGDRrEjBkzshaLe+21V0HHBgAAtU1RBOlDhw6NW2+9Nc4+++xo3LhxHHbYYXHbbbdF375949VXXy308AAAoOhcd9118fLLL8cGG2wQ3377bRx++OFlbV3ShKMAAECRtXaZNGlSbLzxxtnXyy67bFaVnuyzzz5x8cUXF3h0AABQfH7yk59kE43ec8898c4772TV6L169YojjjgiWrRoUejhAQBArdK4WA7yJ06cGKuuumqsueaa8eSTT8bmm28eb7zxRjRr1qzQwwMAgKKUzvY88sgjCz0MAACo9YoiSD/ggANixIgR0bVr1+jdu3d2sD9o0KBs4tGzzjqr0MMDAICic+edd+Zef/TRR9fYWAAAoLYriiD9qquuKvs6TTi62mqrxSuvvBJrr7127LvvvgUdGwAAFKMzzjijwuW5c+fGrFmzomnTptGyZUtBOgAAFFuQniY/at68ednlbbbZJlsAAICl8/XXXy+07sMPP4xTTjklzj333IKMCQAAaquGUQTatm0bxxxzTDz11FMxf/78Qg8HAADqpHTGZzobdMFqdQAAqO+KIki/4447stNM99tvv1hllVXizDPPjDfffLPQwwIAgDo5Aelnn31W6GEAAECtUjSTjablm2++ib/85S9x9913Z61d1lhjjWzi0b59+xZ6iAAAUFQeeeSRCpdLSkpi4sSJ8fvf/z623377go0LAABqo6II0kstt9xycdxxx2XLv//97zjiiCPi0ksvFaQDAMAS2n///StcbtCgQay88sqx6667xnXXXVewcQEAQG1UVEF6mnQ0Vc4MGzYshg8fHu3atTMREgAALAVzDwEAQB0L0p944oksPH/ooYeyno0HH3xwPPnkk7HTTjsVemgAAAAAANRxRRGkp/7o++yzT9x5552x1157RZMmTQo9JAAAKGp9+vSp8rYDBgyo1rEAAEBtVxRB+uTJk7P+6AAAwI/jrbfeypa5c+fGuuuum6374IMPolGjRrH55ptX6J0OAAD1Xa0N0qdPnx7LL7989nVJSUl2eVFKtwMAAKpm3333zYpV7rjjjlhhhRWydV9//XUcd9xxseOOO8bZZ59d6CECAECtUWuD9HQwP3HixGjbtm20bt260kqYFLCn9fPmzSvIGIH66aq3viz0EGCxLthspUIPAajlrrvuumzeodIQPUlfX3HFFbH77rsL0gEAoBiC9GeeeSbatGlT9rVTSgEA4MeTzvj84osvFlqf1n3zzTcFGRMAANRWtTZI33nnncu+7tatW0HHAgAAdc0BBxyQtXFJlelbb711tu61116Lc889Nw488MBCDw8AAGqVWhukl7f22mvHEUcckS3pawAA4IcZOHBgnHPOOXH44YdnE44mjRs3jl69esW1115b6OEBAECt0jCKwC9/+ct47LHHYr311outttoqfve738WkSZMKPSwAAChaLVu2jD/84Q/x1VdfxVtvvZUtU6ZMydYts8wyhR4eAADUKkURpJ911lnxxhtvxHvvvRd77bVX3HzzzdGpU6dsEqQ777yz0MMDAICiNXHixGxJZ36mAL2kpKTQQwIAgFqnKIL0Uuuss05ceuml8cEHH8SLL76YTYSU+joCAABLJlWi77bbbtkxdipWSWF6klq7nH322YUeHgAA1CpFFaQnr7/+epx55pnZ5EgpUD/kkEMKPSQAACg66azPJk2axPjx47M2L6V69uwZw4cPL+jYAACgtimKyUZTYD506NC4++67Y9y4cbHrrrvG1VdfHQceeGAsu+yyhR4eAAAUnSeffDKeeOKJ+MlPflJhfWrx8sknnxRsXAAAUBsVRZBeOsnoqaeeGoceemi0a9eu0EMCAICiNnPmzAqV6KXShKPNmjUryJgAAKC2qvWtXebNmxd//OMfs9NLzzjjDCE6AAD8CHbccce48847yy43aNAg5s+fH9dcc03ssssuBR0bAADUNrW+Ir1Ro0bRu3fv6N69e6ywwgqFHg4AANQJKTBPk42++eabMWfOnDjvvPPi3XffzSrSX3755UIPDwAAapVaX5GebLTRRvHxxx8XehgAAFBnpGPsNBfRDjvsEPvtt1/W6iXNQfTWW2/FmmuuWejhAQBArVLrK9KTK664Is4555y4/PLLY4sttohlllmmwvXLL798wcYGAADFZu7cubHHHnvEwIED49e//nWhhwMAALVeUQTpe+21V/b/z372s6x3Y6mSkpLscuqjDgAAVE2TJk3inXfeKfQwAACgaBRFkP7ss88WeggAAFCnHHnkkTFo0KC46qqrCj0UAACo9YoiSN95550LPQQAAKhTvvvuuxg8eHA8/fTTlbZPHDBgQMHGBgAAtU1RBOkvvPBC7vU77bRTjY0FAACK2ccffxydO3eOf/3rX7H55ptn69Kko+WVb6cIAAAUSZDerVu3hdaVP7jXIx0AAKpm7bXXjokTJ5a1T+zZs2fceOON0a5du0IPDSi0YT1/8F30njx1iW9zU7srfvDjAkB1axhF4Ouvv66wfP755zF8+PDYaqut4sknnyz08AAAoGiUlJRUuPz3v/89Zs6cWbDxAABAMSiKivRWrVottO6nP/1pNG3aNPr06ROjRo0qyLgAAKCuBesAAECRVqQvSjr9dMyYMYUeBgAAFI3UInHBHuh6ogMAQB2oSH/nnXcWqppJfR2vuuqq6NKlS8HGBQAAxSYdSx977LHRrFmz7PK3334bJ598ciyzzDIVtnvggQcKNEIAAKh9iiJIT2F5qpJZ8LTTbbbZJgYPHlywcQEAQLE55phjKlw+8sgjCzYWAAAoFkURpI8bN67C5YYNG8bKK68czZs3L9iYAACgGN1+++2FHgIAABSdogjSV1tttUIPAQAAAACAeqpWTzY6cuTIePTRRyusu/POO2P11VePtm3bxkknnRSzZ88u2PgAAAAAAKj7anWQftlll8W7775bdvmf//xn9OrVK7p37x4XXHBB/O1vf4v+/fsXdIwAAAAAANRttTpIf/vtt2O33XYru3zPPfdE165d49Zbb40+ffrEjTfeGPfdd19BxwgAAAAAQN1Wq4P0r7/+Otq1a1d2+fnnn48999yz7PJWW20VEyZMKNDoAAAAAACoD2p1kJ5C9HHjxmVfz5kzJ0aPHh3bbLNN2fXffPNNNGnSpIAjBAAAAACgrqvVQfpee+2V9UJ/8cUX48ILL4yWLVvGjjvuWHb9O++8E2uuuWZBxwgAAAAAQN3WOGqxyy+/PA488MDYeeedY9lll4077rgjmjZtWnb94MGDY/fddy/oGAEAAAAAqNtqdUX6SiutFC+88ELWKz0tBxxwQIXr77///ujXr99S3ffNN98cnTt3jubNm2cTmL7++utVul2a8LRBgwax//77L9XjAgAAAABQXGp1kF6qVatW0ahRo4XWt2nTpkKFelXde++90adPnyyET33XN9100+jRo0d8/vnnubf7z3/+E+ecc06F9jIAAMCipeKVVIiy4HLqqadm13fr1m2h604++eRCDxsAAIovSP+xDRgwIE488cQ47rjjYoMNNoiBAwdm/ddTq5hFmTdvXhxxxBFx6aWXxhprrFGj4wUAgGL1xhtvxMSJE8uWp556Klt/yCGHlG2Tjs3Lb3PNNdcUcMQAAFBkPdKrw5w5c2LUqFHZ5KWlGjZsGN27d4+RI0cu8naXXXZZtG3bNnr16pVNfppn9uzZ2VJq+vTpP9LoAQCguKy88soVLl911VWx5pprZvMglUpFLe3bt6/yfTreBgCgptW7ivQvv/wyqy5v165dhfXp8qRJkyq9zUsvvRSDBg2KW2+9tUqP0b9//6wdTenSqVOnH2XsAABQ7EUtd911Vxx//PFZC5dSQ4cOzeZH2mijjbKCl1mzZuXej+NtAABqWr0L0pfUN998E0cddVQWoqeD+6pIB//Tpk0rWyZMmFDt4wQAgNruoYceiqlTp8axxx5btu7www/PwvVnn302O47+85//HEceeWTu/TjeBgCgptW71i4pDE8Tl06ePLnC+nS5stNJP/roo2yS0X333bds3fz587P/GzduHGPGjMlOTS2vWbNm2QIAAHwvneW55557RseOHcvWnXTSSWVfb7zxxtGhQ4fYbbfdsuPwBY+zSzneBgCgptW7ivSmTZvGFltsESNGjKgQjKfL22677ULbr7feevHPf/4z3n777bLlZz/7Weyyyy7Z104jBQCAxfvkk0/i6aefjhNOOCF3u65du2b/jx07toZGBgAAi1fvKtKTPn36xDHHHBNbbrllbL311nHDDTfEzJkz47jjjsuuP/roo2OVVVbJei82b94869VYXuvWrbP/F1wPAABU7vbbb4+2bdvG3nvvnbtdKlZJUmU6AADUFvUySO/Zs2d88cUX0bdv32yC0S5dusTw4cPLJiAdP358NGxY74r1AQCgWqQzQFOQnopZUnvEUql9y7Bhw2KvvfaKFVdcMd55550466yzYqeddopNNtmkoGMGAICo70F6ctppp2VLZZ577rnc2w4ZMqSaRgUAAHVPaumSilWOP/74hdouputKzxBNbRMPOuiguOiiiwo2VgAAqEy9DdIBAICasfvuu0dJSclC61Nw/vzzzxdkTAAAsCT0LwEAAAAAgByCdAAAAAAAyCFIBwAAAACAHIJ0AAAAAADIIUgHAAAAAIAcgnQAAAAAAMghSAcAAAAAgByCdAAAAAAAyCFIBwAAAACAHIJ0AAAAAADIIUgHAAAAAIAcgnQAAAAAAMghSAcAAAAAgByCdAAAAAAAyCFIBwAAAACAHIJ0AAAAAADIIUgHAAAAAIAcgnQAAAAAAMghSAcAAAAAgByCdAAAAAAAyCFIBwAAAACAHIJ0AAAAAADIIUgHAAAAAIAcgnQAAAAAAMghSAcAAAAAgByCdAAAAAAAyCFIBwAAAACAHIJ0AAAAAADIIUgHAAAAAIAcgnQAAAAAAMghSAcAAAAAgByCdAAAAAAAyCFIBwAAAACAHIJ0AAAAAADIIUgHAAAAAIAcgnQAAAAAAMghSAcAAAAAgByCdAAAAAAAyCFIBwAAAACAHIJ0AAAAAADIIUgHAAAAAIAcgnQAAAAAAMghSAcAAAAAgByCdAAAAAAAyNE470oAAACgePUa8kaVt+09eWq1jgUAipmKdAAAAAAAyCFIBwAAAACAHIJ0AAAAAADIIUgHAAAAAIAcgnQAAAAAAMghSAcAAAAAgByCdAAAAAAAyCFIBwAAAACAHIJ0AAAAAADIIUgHAAAAAIAcgnQAAAAAAMghSAcAAAAAgByCdAAAoNpccskl0aBBgwrLeuutV3b9t99+G6eeemqsuOKKseyyy8ZBBx0UkydPLuiYAQBgQY0XWgMAAPAj2nDDDePpp58uu9y48fcfQ84666x47LHH4v77749WrVrFaaedFgceeGC8/PLLBRot9VGvIW/U6OMNOnarGn08AOCHE6QDAADVKgXn7du3X2j9tGnTYtCgQTFs2LDYdddds3W33357rL/++vHqq6/GNttsU4DRAgDAwrR2AQAAqtWHH34YHTt2jDXWWCOOOOKIGD9+fLZ+1KhRMXfu3OjevXvZtqnty6qrrhojR45c5P3Nnj07pk+fXmEBAIDqJEgHAACqTdeuXWPIkCExfPjwuOWWW2LcuHGx4447xjfffBOTJk2Kpk2bRuvWrSvcpl27dtl1i9K/f/+sDUzp0qlTpxr4TgAAqM+0dgEAAKrNnnvuWfb1JptskgXrq622Wtx3333RokWLpbrPCy+8MPr06VN2OVWkC9MBAKhOKtIBAIAak6rP11lnnRg7dmzWN33OnDkxderUCttMnjy50p7qpZo1axbLL798hQUAAKqTIB0AAKgxM2bMiI8++ig6dOgQW2yxRTRp0iRGjBhRdv2YMWOyHurbbrttQccJAADlae0CAABUm3POOSf23XffrJ3LZ599Fv369YtGjRrFYYcdlvU379WrV9ampU2bNlllee/evbMQfZtttin00AEAoIwgHQAAqDaffvppFpp/9dVXsfLKK8cOO+wQr776avZ1cv3110fDhg3joIMOitmzZ0ePHj3iD3/4Q6GHDdWq15A3Cj0EAGAJCdIBAIBqc8899+Re37x587j55puzBQAAais90gEAAAAAIIcgHQAAAAAAcgjSAQAAAAAghyAdAAAAAAByCNIBAAAAACBH47wrAQCq21VvfVnoIUCVXLDZSoUeAgAAUCAq0gEAAAAAIIcgHQAAAAAAcgjSAQAAAAAghyAdAAAAAAByCNIBAAAAACCHIB0AAAAAAHII0gEAAAAAIIcgHQAAAAAAcgjSAQAAAAAghyAdAAAAAAByCNIBAAAAACCHIB0AAAAAAHII0gEAAAAAIIcgHQAAAAAAcgjSAQAAAAAghyAdAAAAAAByCNIBAAAAACBHvQ3Sb7755ujcuXM0b948unbtGq+//voit7311ltjxx13jBVWWCFbunfvnrs9AAAAAAB1R70M0u+9997o06dP9OvXL0aPHh2bbrpp9OjRIz7//PNKt3/uuefisMMOi2effTZGjhwZnTp1it133z3++9//1vjYAQAAAACoWfUySB8wYECceOKJcdxxx8UGG2wQAwcOjJYtW8bgwYMr3X7o0KHxy1/+Mrp06RLrrbde3HbbbTF//vwYMWJEpdvPnj07pk+fXmEBAAAAAKA41bsgfc6cOTFq1KisPUuphg0bZpdTtXlVzJo1K+bOnRtt2rSp9Pr+/ftHq1atypZUwQ4AAAAAQHGqd0H6l19+GfPmzYt27dpVWJ8uT5o0qUr3cf75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We/F5Iwzzoidd945rrvuuth7773jnnvuiTfffDP+9Kc/Ffg7AQCA+nGMDgAAhVQvg/SePXvGF198EX379s0mDO3SpUsMHz68bHKj8ePHR8OG3xfrb7fddjFs2LC46KKL4le/+lWsvfba8dBDD8VGG230o40pnZrar1+/hU5RpfbxXBUHz1Px8FwVD89VcfA8FQ/P1ZIdo9cWnreqs6+qzr5aMvZX1dlXVWdfVZ19VXX2Vd3aVw1KUkd1AAAAAACgUvWuRzoAAAAAACwJQToAAAAAAOQQpAMAAAAAQA5BOgAAAAAA5BCk16Cbb745OnfuHM2bN4+uXbvG66+/vshthwwZEg0aNKiwpNtRvV544YXYd999o2PHjtk+f+ihhxZ7m+eeey4233zzbFbhtdZaK3vuqH3PVXqeFnxPpWXSpEk1Nub6qH///rHVVlvFcsstF23bto39998/xowZs9jb3X///bHeeutlP/c23njjePzxx2tkvPXZ0jxXflcVxi233BKbbLJJLL/88tmy7bbbxt///vfc23hP1f7nyfupbh7T1xeLOy4rKSmJvn37RocOHaJFixbRvXv3+PDDD6M+qsrv22+//TZOPfXUWHHFFWPZZZeNgw46KCZPnhz1zeJ+jtpPi3bVVVdl78UzzzyzbJ399X8uueSShX7fpmOkUvZTRf/973/jyCOPzPZH+vmdjiPffPPNsuv9fP8/6bigsrwhvZYSr6vvzZs3Ly6++OJYffXVs9fMmmuuGZdffnn2WiqG15UgvYbce++90adPn+jXr1+MHj06Nt100+jRo0d8/vnni7xNOliYOHFi2fLJJ5/U6Jjro5kzZ2bPTfqAVBXjxo2LvffeO3bZZZd4++23swOVE044IZ544olqH2t9t6TPVan0QaX8+yp9gKH6PP/889kBw6uvvhpPPfVUzJ07N3bffffs+VuUV155JQ477LDo1atXvPXWW9kHzLT861//qtGx1zdL81wlflfVvJ/85CfZB+RRo0ZlH2R23XXX2G+//eLdd9+tdHvvqeJ4nhLvp7p5TF8fLO647Jprrokbb7wxBg4cGK+99loss8wy2X5LwUJ9U5Xft2eddVb87W9/y/4Imrb/7LPP4sADD4z6ZnE/R+2nyr3xxhvxxz/+MfsjRHn21/c23HDDCr9vX3rppbLr7Kfvff3117H99ttHkyZNsj9i/fvf/47rrrsuVlhhhbJt/Hz//n1X/jWVfr4nhxxySPa/19X3rr766uwPpb///e/jvffeyy6n19FNN91UHK+rEmrE1ltvXXLqqaeWXZ43b15Jx44dS/r371/p9rfffntJq1atanCELCi9PR588MHcbc4777ySDTfcsMK6nj17lvTo0aOaR8eSPlfPPvtstt3XX39dY+NiYZ9//nn2PDz//POL3ObnP/95yd57711hXdeuXUt+8Ytf1MAIWZLnyu+q2mOFFVYoue222yq9znuqOJ4n76e6eUxfHy14XDZ//vyS9u3bl1x77bVl66ZOnVrSrFmzkrvvvrukvlvw923aN02aNCm5//77y7Z57733sm1GjhxZUt+V/hy1nyr3zTfflKy99tolTz31VMnOO+9ccsYZZ2Tr7a/v9evXr2TTTTet9Dr7qaLzzz+/ZIcddljk9X6+L1p676255prZPvK6qih9Ljn++OMrrDvwwANLjjjiiKJ4XalIrwFz5szJ/oqeTkUo1bBhw+zyyJEjF3m7GTNmxGqrrRadOnVabAUThZGev/LPa5L+Spb3vFJYXbp0yU4P+ulPfxovv/xyoYdT70ybNi37v02bNovcxvuqeJ6rxO+qwp8aec8992SVjOmU98p4TxXH85R4P9XNY/r6Lp3BmVrpld9vrVq1ytri2G8L/75Nr7FUpV5+f6W2E6uuumq93l8L/hy1nyqXznZIZ0wv+Hvf/qootYhIrajWWGONOOKII2L8+PHZevupokceeSS23HLLrKo6ncm92Wabxa233lp2vZ/viz5euOuuu+L444/P2rt4XVW03XbbxYgRI+KDDz7ILv/jH//IzgrZc889i+J1JUivAV9++WX2i79du3YV1qfLi+rPvO6668bgwYPj4Ycfzt6A8+fPz15sn376aQ2NmqpIz19lz+v06dPjf//7X8HGxcJSeJ5OC/rrX/+aLSmk6NatW3ZaNjUj/RxL7Y/S6YEbbbTREr+v9LOvfc+V31WF889//jPrr5jm5zj55JPjwQcfjA022KDSbb2niuN58n6qm8f0/N/PoMR+q9rv27RPmjZtGq1bt66wbX3dX4v6OWo/LSz9oSF9tkl9+Bdkf30vhXFpXpLhw4dn7SVSaLfjjjvGN998Yz8t4OOPP8720dprr521rz3llFPi9NNPjzvuuCO73s/3yqV5QqZOnRrHHntsdtnrqqILLrggDj300OyPCaltUPoDTfpdmP6oVQyvq8aFHgCVS39lL1+xlD5Irb/++lmvs9SEH1gyKaBIS/n31EcffRTXX399/PnPfy7o2OpThUzqyVy+ByHF/Vz5XVU46edZmpsjVTL+5S9/iWOOOSbrt7iokJba/zx5P0H949ho6X+OUtGECRPijDPOyPoym6g6X2nVa5L6yKdgPZ0Ndt9992WTGlLxj32pIv3KK6/MLqfAM/3MSgVq6b1I5QYNGpS9ztJZDywsvdeGDh0aw4YNy+YrKJ1vMO2vYnhdqUivASuttFI0atRooRl50+X27dtX6T5K/0ozduzYaholSyM9f5U9r2myML+Ea7+tt97ae6qGnHbaafHoo4/Gs88+m00ctTTvq6r+vKTmnqsF+V1Vc1JVy1prrRVbbLFFVnmWJvn73e9+V+m23lPF8TwtyPupbh7T10el+8Z+q9rv27RPUluAVM1YXn3dX4v6OWo/VZRaR6RJjzfffPNo3LhxtqQ/OKTJ+tLXqZLT/qpcqhJeZ511st+3XlcLn9W94B//0x/5S1vh+Pm+sDRR/NNPPx0nnHBC2Tqvq4rOPffcsqr0jTfeOI466qhsMtbSs2lq++tKkF5Dv/zTL/7UA6j8X/bS5bw+meWl00jTaW3pBxm1R3r+yj+vSaoCqOrzSmGlv3x6T1WvNOdY+qCYTsN95plnYvXVV1/sbbyvCmNpnqsF+V1VOOm4Yvbs2ZVe5z1VHM/Tgryf6uYxfX2Ufp+kD77l91tqg/jaa6/Vy/22uN+36TWW/pBWfn+NGTMmC67q4/5a1M9R+6mi3XbbLfudkT7flC6pkji1Sij92v5a9Pwk6Uzl9PvW66qi1HYqff/lpb7WqYI/8fN9YbfffnvWTz7NVVDK66qiWbNmZXPMlJcKFdLP96J4XRV6ttP64p577slmmB0yZEjJv//975KTTjqppHXr1iWTJk3Krj/qqKNKLrjggrLtL7300pInnnii5KOPPioZNWpUyaGHHlrSvHnzknfffbeA30X9mOX8rbfeypb09hgwYED29SeffJJdn56j9FyV+vjjj0tatmxZcu6552azLt98880ljRo1Khk+fHgBv4v6YUmfq+uvv77koYceKvnwww9L/vnPf2azaDds2LDk6aefLuB3UfedcsopJa1atSp57rnnSiZOnFi2zJo1q2ybBX/+vfzyyyWNGzcu+e1vf5u9r/r165fNcp6eN2rXc+V3VWGk5+D5558vGTduXMk777yTXW7QoEHJk08+mV3vPVWcz5P3U904pq+vFndcdtVVV2X76eGHH87eD/vtt1/J6quvXvK///2vpL6pyu/bk08+uWTVVVcteeaZZ0refPPNkm233TZb6pvF/Ry1n/LtvPPO2WeeUvbX/zn77LOz9196XaVjpO7du5estNJKJZ9//nl2vf30vddffz07hvzNb36TfY4eOnRoln/cddddZdv4+f69efPmZa+d888/f6HrvK6+d8wxx5SsssoqJY8++mj2PnzggQey9+B5551XFK8rQXoNuummm7I3TtOmTUu23nrrkldffbXCL7n0Yip15plnlm3brl27kr322qtk9OjRBRp5/fHss89mB/8LLqXPTfo/PVcL3qZLly7Zc7XGGmuU3H777QUaff2ypM/V1VdfXbLmmmtmoUSbNm1KunXrlv0So3pV9hylpfz7ZMGff8l9991Xss4662Tvqw033LDkscceK8Do65elea78riqM448/vmS11VbL9vvKK69csttuu5WFCon3VHE+T95PdeOYvr5a3HHZ/PnzSy6++OLstZ3+EJHeD2PGjCmpj6ry+zYFBb/85S9LVlhhhSy0OuCAA7Kwvb5Z3M9R+2nJgnT76//07NmzpEOHDtnrKoV56fLYsWPLrrefKvrb3/5WstFGG2U/u9dbb72SP/3pTxWu9/P9e6kgIv08r+z797r63vTp07OfTelYKuUzKUf79a9/XTJ79uyieF01SP8UuioeAAAAAABqKz3SAQAAAAAghyAdAAAAAAByCNIBAAAAACCHIB0AAAAAAHII0gEAAAAAIIcgHQAAAAAAcgjSAQAAAAAghyAdAAAAAAByCNIBqLeOPfbY2H///Qs9DAAAAKCWE6QDFFHo26BBg2xp2rRprLXWWnHZZZfFd999V+ih1UolJSXxpz/9Kbp27RrLLrtstG7dOrbccsu44YYbYtasWYUeHgAA/GAjR46MRo0axd57713ooQDUeYJ0gCKyxx57xMSJE+PDDz+Ms88+Oy655JK49tpro76aM2fOIq876qij4swzz4z99tsvnn322Xj77bfj4osvjocffjiefPLJGh0nAABUh0GDBkXv3r3jhRdeiM8++6zQ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",
      "text/plain": [
       "<Figure size 1500x1200 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Cell 13: Visualization and Insights\n",
    "# Visualize feature importance and survival patterns\n",
    "\n",
    "# Feature importance\n",
    "feature_importance = pd.DataFrame({\n",
    "    'feature': X.columns,\n",
    "    'importance': best_model.feature_importances_\n",
    "}).sort_values('importance', ascending=False)\n",
    "\n",
    "print(\"Feature Importance:\")\n",
    "print(feature_importance)\n",
    "\n",
    "# Create visualizations\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
    "\n",
    "# 1. Feature Importance\n",
    "axes[0, 0].barh(feature_importance['feature'], feature_importance['importance'])\n",
    "axes[0, 0].set_title('Feature Importance')\n",
    "axes[0, 0].set_xlabel('Importance')\n",
    "\n",
    "# 2. Survival Rate by Gender\n",
    "survival_by_sex = train_df.groupby('Sex')['Survived'].mean()\n",
    "axes[0, 1].bar(['Male', 'Female'], survival_by_sex.values, color=['lightblue', 'lightcoral'])\n",
    "axes[0, 1].set_title('Survival Rate by Gender')\n",
    "axes[0, 1].set_ylabel('Survival Rate')\n",
    "axes[0, 1].set_ylim(0, 1)\n",
    "\n",
    "# 3. Survival Rate by Passenger Class\n",
    "survival_by_class = train_df.groupby('Pclass')['Survived'].mean()\n",
    "axes[1, 0].bar(survival_by_class.index, survival_by_class.values, color='skyblue')\n",
    "axes[1, 0].set_title('Survival Rate by Passenger Class')\n",
    "axes[1, 0].set_xlabel('Passenger Class')\n",
    "axes[1, 0].set_ylabel('Survival Rate')\n",
    "axes[1, 0].set_ylim(0, 1)\n",
    "\n",
    "# 4. Age Distribution by Survival\n",
    "axes[1, 1].hist(train_df[train_df['Survived']==0]['Age'], alpha=0.7, label='Not Survived', bins=20)\n",
    "axes[1, 1].hist(train_df[train_df['Survived']==1]['Age'], alpha=0.7, label='Survived', bins=20)\n",
    "axes[1, 1].set_title('Age Distribution by Survival')\n",
    "axes[1, 1].set_xlabel('Age')\n",
    "axes[1, 1].set_ylabel('Frequency')\n",
    "axes[1, 1].legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "77fc4360",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "============================================================\n",
      "TITANIC SURVIVAL PREDICTION - SUMMARY\n",
      "============================================================\n",
      "\n",
      "📊 Dataset Information:\n",
      "Training samples: 891\n",
      "Test samples: 418\n",
      "Features used: 7\n",
      "\n",
      "🎯 Model Performance:\n",
      "Best validation accuracy: 0.8045\n",
      "Cross-validation score: 0.8301\n",
      "\n",
      "🔍 Key Insights:\n",
      "Overall survival rate: 38.38%\n",
      "Female survival rate: 74.20%\n",
      "Male survival rate: 18.89%\n",
      "Class 1 survival rate: 62.96%\n",
      "Class 2 survival rate: 47.28%\n",
      "Class 3 survival rate: 24.24%\n",
      "\n",
      "🏆 Most Important Features:\n",
      "1. Sex: 0.3533\n",
      "2. Fare: 0.2146\n",
      "3. Age: 0.1838\n",
      "4. Pclass: 0.1302\n",
      "5. SibSp: 0.0461\n",
      "6. Embarked: 0.0369\n",
      "7. Parch: 0.0352\n",
      "\n",
      "📁 Output Files:\n",
      "- titanic_predictions.csv (predictions for test set)\n",
      "\n",
      "✅ Analysis Complete!\n"
     ]
    }
   ],
   "source": [
    "# Cell 14: Summary and Key Insights\n",
    "# Summarize key insights from the data and model results\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"TITANIC SURVIVAL PREDICTION - SUMMARY\")\n",
    "print(\"=\" * 60)\n",
    "\n",
    "print(f\"\\n📊 Dataset Information:\")\n",
    "print(f\"Training samples: {len(train_df)}\")\n",
    "print(f\"Test samples: {len(test_df)}\")\n",
    "print(f\"Features used: {len(X.columns)}\")\n",
    "\n",
    "print(f\"\\n🎯 Model Performance:\")\n",
    "print(f\"Best validation accuracy: {best_accuracy:.4f}\")\n",
    "print(f\"Cross-validation score: {grid_search.best_score_:.4f}\")\n",
    "\n",
    "print(f\"\\n🔍 Key Insights:\")\n",
    "print(f\"Overall survival rate: {train_df['Survived'].mean():.2%}\")\n",
    "\n",
    "# Gender insights\n",
    "female_survival = train_df[train_df['Sex']==1]['Survived'].mean()\n",
    "male_survival = train_df[train_df['Sex']==0]['Survived'].mean()\n",
    "print(f\"Female survival rate: {female_survival:.2%}\")\n",
    "print(f\"Male survival rate: {male_survival:.2%}\")\n",
    "\n",
    "# Class insights\n",
    "for pclass in sorted(train_df['Pclass'].unique()):\n",
    "    class_survival = train_df[train_df['Pclass']==pclass]['Survived'].mean()\n",
    "    print(f\"Class {pclass} survival rate: {class_survival:.2%}\")\n",
    "\n",
    "print(f\"\\n🏆 Most Important Features:\")\n",
    "for i, (feature, importance) in enumerate(zip(feature_importance['feature'], feature_importance['importance']), 1):\n",
    "    print(f\"{i}. {feature}: {importance:.4f}\")\n",
    "\n",
    "print(f\"\\n📁 Output Files:\")\n",
    "print(\"- titanic_predictions.csv (predictions for test set)\")\n",
    "\n",
    "print(\"\\n✅ Analysis Complete!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
