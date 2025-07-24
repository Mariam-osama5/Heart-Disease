import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import GridSearchCV
import joblib

#1-Data Preprocessing + EDA
df = pd.read_csv("https://raw.githubusercontent.com/datamanim/heart-disease-dataset/main/heart.csv")

# EDA
print(df.info())
print(df.describe())
print(df['target'].value_counts())

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('target', axis=1))
y = df['target']

plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()
#--------------------------------------------------------------------
#2-PCA

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print("Number of shapes", X_pca.shape[1])

plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.grid()
plt.show()
#--------------------------------------------------------------------
#Feature Selection

# RandomForest Feature Importance
model = RandomForestClassifier()
model.fit(X_scaled, y)
importances = pd.Series(model.feature_importances_, index=df.drop('target', axis=1).columns)
importances.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importances (Random Forest)")
plt.show()

# RFE
rfe = RFE(model, n_features_to_select=8)
rfe.fit(X_scaled, y)
selected = df.drop('target', axis=1).columns[rfe.support_]
print("Selected features by RFE:", selected.tolist())
#--------------------------------------------------------------------
#supervised Learning

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print(f"🔹 {name}")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("-" * 40)
#--------------------------------------------------------------------
#Clustering (KMeans + Hierarchical)

# KMeans
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# Hierarchical
linked = linkage(X_scaled, 'ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=10)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()
#--------------------------------------------------------------------
#Hyperparameter Tuning

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 3, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parametars:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)
#--------------------------------------------------------------------
#save

joblib.dump(grid.best_estimator_, "final_model.pkl")

