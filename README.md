\# 🏢 Real Estate Buyer Segmentation \& Investment Profiling



Machine Learning–based buyer segmentation project built for real estate market intelligence.



This project analyzes buyer behavior using clustering techniques and provides an interactive Streamlit dashboard for exploring customer segments, investment patterns, and portfolio characteristics.



\---



\## 📌 Project Overview



Real estate companies often possess large volumes of customer and transaction data but struggle to identify meaningful buyer groups.



This project applies:



\* Data Cleaning \& Feature Engineering

\* Exploratory Data Analysis (EDA)

\* K-Means Clustering

\* Hierarchical Clustering

\* Cluster Validation

\* Interactive Dashboard Development



to uncover hidden buyer segments and support data-driven decision-making.



\---



\## 🚀 Dashboard Preview



\### Overview Dashboard



!\[Dashboard Overview](assets/dashboard-home.png)



\### Segment Explorer



!\[Segment Explorer](assets/segment-explorer.png)



\### Methodology \& Validation



!\[Methodology](assets/methodology.png)



\---



\## 📊 Dataset



The project uses two connected datasets:



\### Clients Dataset



Contains buyer information:



\* Client ID

\* Age

\* Country

\* Region

\* Client Type

\* Satisfaction Score

\* Loan Status

\* Acquisition Purpose

\* Referral Channel



\### Properties Dataset



Contains property transaction information:



\* Listing ID

\* Client Reference

\* Sale Price

\* Floor Area

\* Unit Category

\* Transaction Date

\* Tower Number

\* Listing Status



The datasets are connected through:



```text

client\_id  ←→  client\_ref

```



\---



\## ⚙️ Data Pipeline



\### 1. Data Cleaning



\* Mixed date format correction

\* Currency value cleaning

\* Missing value handling

\* Duplicate removal



\### 2. Feature Engineering



Created client-level behavioral features:



\* Number of Properties Owned

\* Total Spend

\* Average Ticket Size

\* Investment Tenure

\* Tenure per Property

\* Office Purchase Ratio

\* Loan Usage Flag

\* Company Buyer Flag



\### 3. Clustering



Algorithms used:



\* K-Means Clustering

\* Agglomerative Hierarchical Clustering



Validation methods:



\* Elbow Method

\* Silhouette Score

\* Adjusted Rand Index (ARI)



\---



\## 🎯 Final Buyer Segments



\### 🐢 Slow-Accumulating Buyers



Clients who build their portfolio gradually over time.



\### 🐇 Fast-Accumulating Buyers



Clients who acquire properties within shorter time windows.



\### 🏦 Loan-Backed Buyers



Clients who rely on financing for purchases.



\### 🏢 Corporate Buyers



Organizations purchasing property as business entities.



\### 📦 Large-Portfolio Buyers



High-volume buyers owning significantly more properties than the average client.



\---



\## 📈 Key Findings



\### Findings Supported by Data



✅ Large portfolio owners form a distinct segment.



✅ Loan-backed buyers form a separate behavioral group.



✅ Corporate buyers can be isolated clearly.



✅ Acquisition pace is an important differentiator.



\### Findings Not Supported by Data



❌ Younger buyers were not significantly more loan dependent.



❌ Investment-purpose buyers did not spend significantly more.



❌ Companies did not purchase significantly more units.



❌ Satisfaction score was not linked to spending behavior.



\---



\## 🖥️ Technology Stack



\### Data Science



\* Python

\* Pandas

\* NumPy

\* Scikit-Learn



\### Visualization



\* Plotly

\* Streamlit



\### Development



\* Git

\* GitHub

\* VS Code



\---



\## 📂 Project Structure



```text

real-estate-segmentation/

│

├── app/

│   └── dashboard.py

│

├── src/

│   ├── preprocessing.py

│   └── clustering.py

│

├── data/

│   ├── clients.csv

│   └── properties.csv

│

├── outputs/

│   ├── clustered\_clients.csv

│   ├── k\_evaluation.csv

│   └── models/

│

├── assets/

│   ├── dashboard-home.png

│   ├── segment-explorer.png

│   └── methodology.png

│

├── requirements.txt

└── README.md

```



\---



\## ▶️ Run Locally



Clone the repository:



```bash

git clone https://github.com/<your-username>/real-estate-segmentation.git

cd real-estate-segmentation

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Generate features and clusters:



```bash

python src/preprocessing.py

python src/clustering.py

```



Launch dashboard:



```bash

streamlit run app/dashboard.py

```



\---



\## 📚 Learning Outcomes



This project demonstrates:



\* End-to-end machine learning workflow

\* Customer segmentation

\* Feature engineering

\* Cluster validation

\* Data storytelling

\* Dashboard development

\* Model deployment



\---



\## 👨‍💻 Author



\*\*Srijan\*\*



B.Tech Computer Science Engineering

GITAM University





