import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading the data

import pandas as pd

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(train_df.head())

print(train_df.info())


# Drop columns with too many missing values or not useful
train_df = train_df.drop(columns=["Cabin", "Ticket", "Name"])

# Fill missing Age with median
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)

# Fill missing Embarked with mode
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)

# Confirm no more missing values and show how many missing values exist in each column
print(train_df.isnull().sum())


train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
train_df["Embarked"] = train_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})


# Separate features (X) and target variable (y)
X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_df['Survived']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFeatures used:")
print(X.columns.tolist())

#Data Splitting and Scaling
# Split the training data into training and validation sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\nData scaled successfully!")


# Train Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_val_scaled)

# Evaluate model
accuracy = accuracy_score(y_val, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Cell 10: Hyperparameter Tuning
# Tune hyperparameters to improve accuracy
from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                          param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get best model
best_model = grid_search.best_estimator_
best_accuracy = accuracy_score(y_val, best_model.predict(X_val_scaled))

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Best validation accuracy: {best_accuracy:.4f}")

# Cell 11: Predict on Test Data
# Apply the selected model on the test dataset
# First, clean and preprocess test data similar to training data

# Clean test data
test_df_clean = test_df.drop(columns=['Cabin', 'Ticket', 'Name'])

# Fill missing values in test data
test_df_clean['Age'].fillna(test_df_clean['Age'].median(), inplace=True)
test_df_clean['Fare'].fillna(test_df_clean['Fare'].median(), inplace=True)
test_df_clean['Embarked'].fillna(test_df_clean['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
test_df_clean['Sex'] = test_df_clean['Sex'].map({'male': 0, 'female': 1})
test_df_clean['Embarked'] = test_df_clean['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Prepare test features
X_test = test_df_clean[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Scale test features
X_test_scaled = scaler.transform(X_test)

# Make predictions
test_predictions = best_model.predict(X_test_scaled)

# Create submission dataframe
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

print("Test predictions completed!")
print(f"Predicted survivors: {sum(test_predictions)}")
print(f"Predicted non-survivors: {len(test_predictions) - sum(test_predictions)}")

# Cell 12: Save Predictions
# Save predictions to CSV file
submission.to_csv('titanic_predictions.csv', index=False)
print("Predictions saved to 'titanic_predictions.csv'")
print("\nFirst 10 predictions:")
print(submission.head(10))

# Cell 13: Visualization and Insights
# Visualize feature importance and survival patterns

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature Importance
axes[0, 0].barh(feature_importance['feature'], feature_importance['importance'])
axes[0, 0].set_title('Feature Importance')
axes[0, 0].set_xlabel('Importance')

# 2. Survival Rate by Gender
survival_by_sex = train_df.groupby('Sex')['Survived'].mean()
axes[0, 1].bar(['Male', 'Female'], survival_by_sex.values, color=['lightblue', 'lightcoral'])
axes[0, 1].set_title('Survival Rate by Gender')
axes[0, 1].set_ylabel('Survival Rate')
axes[0, 1].set_ylim(0, 1)

# 3. Survival Rate by Passenger Class
survival_by_class = train_df.groupby('Pclass')['Survived'].mean()
axes[1, 0].bar(survival_by_class.index, survival_by_class.values, color='skyblue')
axes[1, 0].set_title('Survival Rate by Passenger Class')
axes[1, 0].set_xlabel('Passenger Class')
axes[1, 0].set_ylabel('Survival Rate')
axes[1, 0].set_ylim(0, 1)

# 4. Age Distribution by Survival
axes[1, 1].hist(train_df[train_df['Survived']==0]['Age'], alpha=0.7, label='Not Survived', bins=20)
axes[1, 1].hist(train_df[train_df['Survived']==1]['Age'], alpha=0.7, label='Survived', bins=20)
axes[1, 1].set_title('Age Distribution by Survival')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Training data shape:", train_df.shape)
print("Test data shape:    ", test_df.shape)


# Cell 14: Summary and Key Insights
# Summarize key insights from the data and model results

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION - SUMMARY")
print("=" * 60)

print(f"\n📊 Dataset Information:")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Features used: {len(X.columns)}")

print(f"\n🎯 Model Performance:")
print(f"Best validation accuracy: {best_accuracy:.4f}")
print(f"Cross-validation score: {grid_search.best_score_:.4f}")

print(f"\n🔍 Key Insights:")
print(f"Overall survival rate: {train_df['Survived'].mean():.2%}")

# Gender insights
female_survival = train_df[train_df['Sex']==1]['Survived'].mean()
male_survival = train_df[train_df['Sex']==0]['Survived'].mean()
print(f"Female survival rate: {female_survival:.2%}")
print(f"Male survival rate: {male_survival:.2%}")

# Class insights
for pclass in sorted(train_df['Pclass'].unique()):
    class_survival = train_df[train_df['Pclass']==pclass]['Survived'].mean()
    print(f"Class {pclass} survival rate: {class_survival:.2%}")

print(f"\n🏆 Most Important Features:")
for i, (feature, importance) in enumerate(zip(feature_importance['feature'], feature_importance['importance']), 1):
    print(f"{i}. {feature}: {importance:.4f}")

print(f"\n📁 Output Files:")
print("- titanic_predictions.csv (predictions for test set)")

print("\n✅ Analysis Complete!")