# necessary imports
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# dataset is accessed via 'df'
df = pd.read_csv("medical_insurance.csv")

# eliminating the outliers using IQR rule
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)

IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_filtered = df[(df['bmi'] >= lower_bound) & (df['bmi'] < 47)]  # upper bound = 47 removes all outliers

# renaming the filtered dataset for easier management
df = df_filtered

# splitting the feature and target variables
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# initialising the label encoders
labEnc_sex = LabelEncoder()
labEnc_smoker = LabelEncoder()
labEnc_region = LabelEncoder()

# label encoding categorical columns
x[:, 1] = labEnc_sex.fit_transform(x[:, 1])  # Gender: Female 0, Male 1
x[:, 4] = labEnc_smoker.fit_transform(x[:, 4])  # Smoker: Yes 1, No 0
x[:, 5] = labEnc_region.fit_transform(x[:, 5])  # Region: Northeast 0, Northwest 1, Southeast 2, Southwest 3

# initialising ColumnTransformer with OneHotEncoder algorithm
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), [1, 4, 5])  # Dropping the dummy variable column
    ],
    remainder='passthrough'
)

# fitting and transforming the feature set
x = ct.fit_transform(x)

# splitting the training data and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=2
)

# creating an instance of StandardScaler instance and fitting and transforming the feature set
ss_x = StandardScaler()
scaled_x_train = ss_x.fit_transform(x_train)
scaled_x_test = ss_x.transform(x_test)

label_encoders = {
    'sex': labEnc_sex,
    'smoker': labEnc_smoker,
    'region': labEnc_region
}

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)


with open('column_transformer.pkl', 'wb') as f:
    pickle.dump(ct, f)

# defining a function to get the inputs from the user
def get_input():
    age = int(input("Enter the age: "))
    bmi = float(input("Enter BMI: "))
    children = int(input("Enter no. of children: "))
    
    # infinite loops break only if valid inputs are received
    while True:
        print("\nSelect Gender")
        print("1: Female")
        print("2: Male")
        gender_choice = input("Enter either 1 or 2: ")
        if gender_choice == '1':
            gender = 'female'
            break
        elif gender_choice == '2':
            gender = 'male'
            break
        else:
            print("Invalid input. Please enter 1 for Female or 2 for Male.")

    while True:
        print("\nAre you a smoker?")
        print("1: Yes")
        print("2: No")
        smoker_choice = input("Enter 1 for Yes or 2 for No: ")
        if smoker_choice == '1':
            smoker = 'yes'
            break
        elif smoker_choice == '2':
            smoker = 'no'
            break
        else:
            print("Invalid input. Please enter 1 for Yes or 2 for No.")
    
    while True:
        print("\nSelect region:")
        print("1: Northeast")
        print("2: Northwest")
        print("3: Southeast")
        print("4: Southwest")
        region_choice = input("Enter 1, 2, 3, or 4 for the corresponding region: ")
        if region_choice == '1':
            region = 'northeast'
            break
        elif region_choice == '2':
            region = 'northwest'
            break
        elif region_choice == '3':
            region = 'southeast'
            break
        elif region_choice == '4':
            region = 'southwest'
            break
        else:
            print("Invalid input. Please enter a number from 1 to 4 for the region.")

    return np.array([[age, gender, bmi, children, smoker, region]], dtype=object)

# calling the function to get user input
x_user = get_input()
cols = ['Age', 'Gender', 'BMI', 'No. of Children', 'Smoker', 'Region']
userDF = pd.DataFrame(x_user, columns=cols)
print('\nUser Input Data\n', userDF.to_string(index=False))

# Load the pickled encoders and column transformer
with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
    labEnc_sex = encoders['sex']
    labEnc_smoker = encoders['smoker']
    labEnc_region = encoders['region']


with open('column_transformer.pkl', 'rb') as f:
    ct = pickle.load(f)

# encoding and transforming the input values
x_user[:, 1] = labEnc_sex.transform(x_user[:, 1])
x_user[:, 4] = labEnc_smoker.transform(x_user[:, 4])
x_user[:, 5] = labEnc_region.transform(x_user[:, 5])
x_user = ct.transform(x_user)

# the hyperparameters are obtained after HyperParameter Tuning using GridSearchCV
with open('rf_model.pkl','rb') as f:
    rfr=pickle.load(f)
    
# predicting the insurance cost
y_user = rfr.predict(x_user)

# printing the prediction
print("Insurance cost prediction for user input data is:", y_user[0].round(2))
