# Practical 7: Customer Segmentation Using K-Means Clustering

# Step 1 - Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 2 - Create Dataset
data = {
    "Customer": ["C1", "C2", "C3", "C4", "C5",
                 "C6", "C7", "C8", "C9",
                 "C10", "C11", "C12"],
    "Annual_Income": [25, 28, 30, 32, 35,
                      55, 58, 60, 62,
                      85, 90, 95],
    "Spending_Score": [20, 25, 18, 30, 28,
                       55, 60, 52, 58,
                       85, 90, 82]
}

# Step 3 - Create DataFrame
df = pd.DataFrame(data)

# Step 4 - Display Dataset
print("Customer Dataset:")
print(df)

# Step 5 - Select Variables for Clustering
# No y: clustering is unsupervised
X = df[["Annual_Income", "Spending_Score"]]

# Step 6 - Calculate Inertia for K = 1 to 5
inertia = []
for k in range(1, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)

# Step 7 - Plot the Elbow Curve
plt.plot(range(1, 6), inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Step 8 - Apply K-Means with K = 3
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Step 9 - Assign Customers to Clusters (+1 so labels are 1, 2, 3)
df["Cluster"] = kmeans.fit_predict(X) + 1

# Step 10 - Display Cluster Assignments
print("Customer Cluster Assignments:")
print(df)

# Step 11 - Display Cluster Centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Step 12 - Visualize the Clusters
plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"]
)
plt.xlabel("Annual Income (Lakhs)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()
