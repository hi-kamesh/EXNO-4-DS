# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
     import pandas as pd
     df1=pd.read_csv("/content/bmi.csv")
     df1

```
<img width="500" height="530" alt="image" src="https://github.com/user-attachments/assets/3672bc59-c19a-4fa7-a6b8-26622d130297" />

```
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,Normalizer,RobustScaler

df2=df1.copy()
enc=StandardScaler()
df2[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df2

```
<img width="734" height="532" alt="image" src="https://github.com/user-attachments/assets/444b82c4-f200-4503-b086-13931dd796bb" />

```
df3=df1.copy()
enc=MinMaxScaler()
df3[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df3
```
<img width="709" height="525" alt="image" src="https://github.com/user-attachments/assets/1fa61cc0-d9ae-46b0-9cda-9c9564c575ba" />

```
df4=df1.copy()
enc=MaxAbsScaler()
df4[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df4

```
<img width="728" height="503" alt="image" src="https://github.com/user-attachments/assets/be3ae04a-0260-4cc4-b835-ebccfed36498" />

```
df5=df1.copy()
enc=Normalizer()
df5[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df5

```
<img width="689" height="534" alt="image" src="https://github.com/user-attachments/assets/5ea13053-ae7d-496c-bfb7-57e7833bab05" />

```

df6=df1.copy()
enc=RobustScaler()
df6[['new_heights','new_weight']]=enc.fit_transform(df2[['Height','Weight']])
df6

```
<img width="701" height="502" alt="image" src="https://github.com/user-attachments/assets/571ebc9f-5524-4a88-942e-b6e5bdfa4132" />

```
import pandas as pd

df=pd.read_csv("/content/income(1) (1).csv")
df
```
<img width="1416" height="628" alt="image" src="https://github.com/user-attachments/assets/1c191f0e-01d9-4d52-ac77-ebfc8565fba0" />

```
from sklearn.preprocessing import LabelEncoder

df_encoded=df.copy()
le=LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
x=df_encoded.drop("SalStat",axis=1)
y=df_encoded["SalStat"]
x

```
<img width="1390" height="557" alt="image" src="https://github.com/user-attachments/assets/b0bdc0a0-fa95-46b5-91d9-fda3bb434fa0" />

```
y
```
<img width="290" height="577" alt="image" src="https://github.com/user-attachments/assets/8af383d1-91e8-44cc-a47c-39d993032dff" />

```
import pandas as pd

from sklearn.feature_selection import SelectKBest, chi2

chi2_selector=SelectKBest(chi2,k=5)
chi2_selector.fit(x,y)

selected_features_chi2=x.columns[chi2_selector.get_support()]
print("Selected features(chi_square):",list(selected_features_chi2))

mi_scores=pd.Series(chi2_selector.scores_,index = x.columns)
print(mi_scores.sort_values(ascending=False))

```
<img width="1043" height="325" alt="image" src="https://github.com/user-attachments/assets/079d6c7f-b6eb-4368-990a-f534fdb73422" />

```
from sklearn.feature_selection import f_classif

anova_selector=SelectKBest(f_classif,k=5)
anova_selector.fit(x,y)

selected_features_anova=x.columns[anova_selector.get_support()]
print("Selected features(chi_square):",list(selected_features_anova))

mi_scores=pd.Series(anova_selector.scores_,index = x.columns)
print(mi_scores.sort_values(ascending=False))

```
<img width="1010" height="328" alt="image" src="https://github.com/user-attachments/assets/6e09d5d6-164c-46ae-bba0-422a60aaac98" />

```
from sklearn.feature_selection import mutual_info_classif

mi_selector=SelectKBest(mutual_info_classif,k=5)
mi_selector.fit(x,y)

selected_features_mi=x.columns[mi_selector.get_support()]
print("Selected features(Mutual Info):",list(selected_features_mi))

mi_scores=pd.Series(anova_selector.scores_,index = x.columns)
print(mi_scores.sort_values(ascending=False))

```
<img width="1114" height="313" alt="image" src="https://github.com/user-attachments/assets/13cfd614-bbb8-4fb2-a44d-a3382848b17b" />

```
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(max_iter=100)
rfe= RFE(model, n_features_to_select=5)
rfe.fit(x,y)

selected_features_rfe = x.columns[rfe.support_]
print("Selected features (RFE):",list(selected_features_rfe))

```
<img width="1201" height="181" alt="image" src="https://github.com/user-attachments/assets/baa09934-e5ce-48d3-bb82-3f0920fb9d2c" />

```
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x,y)

importances = pd.Series(rf.feature_importances_, index=x.columns)
selected_features_rf = importances.sort_values(ascending=False).head(5).index
print(importances)
print("Top 5 features (Random Forest Importance):",list(selected_features_rf))

```
<img width="1271" height="323" alt="image" src="https://github.com/user-attachments/assets/d1b34cca-f3d2-4a0b-89f6-4ea5ad8c2edc" />

```
from sklearn.linear_model import LassoCV
import numpy as np
 
    
lasso=LassoCV(cv=5).fit(x,y)
importance = np.abs(lasso.coef_)

selected_features_lasso=x.columns[importance>0]
print("Selected features (Lasso):",list(selected_features_lasso))

```
<img width="853" height="55" alt="image" src="https://github.com/user-attachments/assets/710de38f-0a22-4f1c-96b3-2721e34eaa18" />

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("/content/income(1) (1).csv")

df_encoded = df.copy()

le = LabelEncoder()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("SalStat", axis=1)
y = df_encoded["SalStat"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)  # you can tune k
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

```

<img width="724" height="369" alt="image" src="https://github.com/user-attachments/assets/2151aa38-1ecb-4679-aa4c-dc95b61ee7a9" />




# RESULT:
       Thus, Feature selection and Feature scaling has been used on thegiven dataset.
