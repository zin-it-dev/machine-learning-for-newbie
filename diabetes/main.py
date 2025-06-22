import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

df = pd.read_csv('../data/diabetes.csv')

# profile = ProfileReport(df, title="Diabetes Report", explorative=True)
# profile.to_file("./reports/diabets.html")

target = "Outcome"
x = df.drop(target, axis=1)
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))