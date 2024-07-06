# HeartDiseaseDiagnosticAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('Heart Disease data.csv')

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# Fill missing values or drop rows/columns
data = data.dropna() 
data.columns = data.columns.str.strip()

# Convert categorical variables to numerical if necessary
data['sex'] = data['sex'].map({1: 'male', 0: 'female'})
data['cp'] = data['cp'].map({0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal pain', 3: 'asymptomatic'})
data['fbs'] = data['fbs'].map({1: '> 120 mg/dl', 0: '<= 120 mg/dl'})
data['restecg'] = data['restecg'].map({0: 'normal', 1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy'})
data['exang'] = data['exang'].map({1: 'yes', 0: 'no'})
data['thal'] = data['thal'].map({0: 'normal', 1: 'fixed defect', 2: 'reversable defect'})

# Any other transformations
# For example, if 'age' needs to be categorized
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 45, 60, 100], labels=['0-30', '31-45', '46-60', '61+'])

# Save the cleaned and transformed data to a new CSV file
data.to_csv('transformed_heart_disease_data.csv', index=False)
print(data.describe())

# Distribution of target variable
sns.countplot(x='target', data=data)
plt.title('Distribution of Heart Disease')
plt.show()

# Check the unique values and data types
print(data['sex'].unique())  # Ensure it's categorical or numerical
print(data['target'].unique())  # Ensure it's categorical or numerical

print(data.columns)
data = pd.DataFrame({'sex':['male','female','male','female','male'],'target':[1,0,1,0,1]})

# Plot heart disease rates by gender
sns.countplot(x='sex', hue='target', data=data)
plt.title('Heart Disease by Gender')
plt.show()

print(data.columns)
data = pd.DataFrame({
    'age': [25, 40, 55, 70, 35],
    'target': [1, 0, 1, 0, 1],
    'age_group': ['0-30', '31-45', '46-60', '61+', '31-45']  # Ensure all entries are strings
})
print(data.columns)

# Plot heart disease rates by age group
sns.countplot(x='age_group', hue='target', data=data)
plt.title('Heart Disease by Age Group')
plt.show()

# Correlation heatmap
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

##Visualization and Dashboard

import plotly.express as px

# Define the DataFrame
data = pd.DataFrame({
    'age': [25, 40, 55, 70, 35],
    'target': [1, 0, 1, 0, 1],
    'serum cholestoral': [200, 240, 180, 220, 260]  # Sample cholesterol levels
})
print(data.columns)

sns.histplot(data=data, x='serum cholestoral', hue='target', kde=True)
plt.title('Cholesterol Levels by Heart Disease Presence')
plt.show()

data = pd.DataFrame({
    'age': [25, 40, 55, 70, 35],
    'target': [1, 0, 1, 0, 1],
    'maximum_heart_rate_achieved': [150, 140, 130, 120, 160]  # Sample max heart rates
})
print(data.columns)

# Scatter plot of age vs. max heart rate
sns.scatterplot(data=data, x='age', y='maximum_heart_rate_achieved', hue='target')
plt.title('Age vs. Max Heart Rate')
plt.show()

# Interactive visualization with Plotly
fig = px.scatter(data, x='age', y='maximum_heart_rate_achieved', color='target', title='Age vs. Max Heart Rate')
fig.show()

import sys
print(sys.executable)

!pip list

!pip install dash

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

print("Dash installed successfully!")

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Sample DataFrame
data = pd.DataFrame({
    'age': [25, 40, 55, 70, 35],
    'target': [1, 0, 1, 0, 1],
    'age_group': ['0-30', '31-45', '46-60', '61+', '31-45'],
    'maximum_heart_rate': [180, 150, 160, 140, 170]
})

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([
    dcc.Checklist(
        id='age-group-filter',
        options=[
            {'label': '0-30', 'value': '0-30'},
            {'label': '31-45', 'value': '31-45'},
            {'label': '46-60', 'value': '46-60'},
            {'label': '61+', 'value': '61+'}
        ],
        value=['0-30', '31-45', '46-60', '61+']
    ),
    dcc.Graph(id='age-maxhr-scatter')
])

# Callback to update graph
@app.callback(
    Output('age-maxhr-scatter', 'figure'),
    Input('age-group-filter', 'value')
)
def update_graph(selected_age_groups):
    filtered_data = data[data['age_group'].isin(selected_age_groups)]
    fig = px.scatter(filtered_data, x='age', y='maximum_heart_rate', color='target')
    return fig

    # Run the server
if __name__ == '__main__':
    app.run_server(debug=True)

    ##KeyMatrics and Relationships

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.drop(['target', 'age_group'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to numerical
y = data['target']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Feature importance
feature_importance = model.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(8, 4))
sns.barplot(x=feature_importance, y=features)
plt.title('Feature Importance')
plt.show()




    



