Retail Customer Segmentation Project
Day-by-Day Work Plan (16 Days)

Team Members & Role Distribution

Person 1: Heramb – Data Engineer & Infrastructure Lead
Primary Focus: Dataset management, data cleaning, preprocessing, and project setup

Person 2: Aayush – Data Analyst & Visualization Lead
Primary Focus: Exploratory data analysis, visualizations, and insights generation

Person 3: Shivam – ML Engineer & Business Strategy Lead
Primary Focus: Clustering implementation, model optimization, and business recommendations

Project Goal

Use behavioral clustering (K-Means) to analyze customer purchasing behavior and segment customers into meaningful groups. These segments will help retailers design personalized marketing strategies and improve customer engagement.

Work Distribution Strategy

Heramb owns the data pipeline from raw data to clean, analysis-ready datasets
Aayush owns the exploratory analysis and all visualizations throughout the project
Shivam owns the machine learning implementation and translating results to business value

------------------------------------------------------------

PHASE 1: PROJECT FOUNDATION (Days 1-3)

DAY 1 – Problem Understanding & Research

Heramb (Data Engineer)
- Research retail personalization challenges and data requirements
- Identify data quality issues commonly found in retail datasets
- Document data pipeline requirements

Aayush (Data Analyst)
- Study customer segmentation concepts and business use cases
- Research visualization techniques for customer behavior analysis
- Identify key metrics for retail customer analysis

Shivam (ML Engineer)
- Research K-Means clustering and unsupervised learning fundamentals
- Study clustering evaluation metrics (silhouette score, elbow method)
- Review business applications of customer segmentation

------------------------------------------------------------

DAY 2 – Dataset Acquisition & Initial Assessment

Heramb (Data Engineer) – LEAD
- Search and download Mall Customer Segmentation Dataset
- Assess data quality and completeness
- Document dataset schema and metadata

Aayush (Data Analyst)
- Review dataset features and their business meanings
- Identify potential relationships between variables
- Plan visualization strategy for EDA

Shivam (ML Engineer)
- Evaluate dataset suitability for clustering
- Identify which features are relevant for behavioral segmentation
- Plan feature engineering requirements

------------------------------------------------------------

DAY 3 – Environment Setup & Project Structure

Heramb (Data Engineer) – LEAD
- Install Anaconda and configure Python 3.10+ environment
- Create project folder structure (data/raw, data/processed, notebooks, src, visuals)
- Set up data loading scripts and utilities

Aayush (Data Analyst)
- Install visualization libraries (Matplotlib, Seaborn)
- Set up Jupyter Notebook environment
- Create visualization templates and style configurations

Shivam (ML Engineer)
- Initialize Git repository and create .gitignore
- Install Scikit-Learn and ML dependencies
- Set up version control workflow and branching strategy

------------------------------------------------------------

PHASE 2: DATA PREPARATION (Days 4-6)

DAY 4 – Dataset Loading & Initial Inspection

Heramb (Data Engineer) – LEAD
- Load dataset using Pandas into notebooks/analysis.ipynb
- Create data loading functions in src/data_loader.py
- Generate initial data quality report

Aayush (Data Analyst)
- Inspect dataset columns, data types, and structure
- Create summary statistics tables
- Document initial observations

Shivam (ML Engineer)
- Check dataset size and memory requirements
- Verify data format compatibility with Scikit-Learn
- Assess computational requirements for clustering

------------------------------------------------------------

DAY 5 – Data Cleaning & Preprocessing

Heramb (Data Engineer) – LEAD
- Handle missing values (imputation or removal strategy)
- Remove duplicate records
- Standardize column names and formats
- Save cleaned dataset to data/processed/

Aayush (Data Analyst)
- Identify outliers and anomalies in the data
- Document data cleaning decisions and their impact
- Validate data distributions after cleaning

Shivam (ML Engineer)
- Verify data consistency and integrity
- Check for data leakage issues
- Ensure cleaned data meets ML requirements

------------------------------------------------------------

DAY 6 – Feature Understanding & Selection

Heramb (Data Engineer)
- Create feature documentation with descriptions
- Generate correlation matrix for numerical features
- Prepare feature subsets for analysis

Aayush (Data Analyst) – LEAD
- Analyze relationships between income, age, spending score
- Create preliminary scatter plots to understand feature interactions
- Document feature importance for business context

Shivam (ML Engineer)
- Identify optimal features for clustering (Annual Income, Spending Score, Purchase Frequency)
- Plan feature scaling/normalization strategy
- Document feature selection rationale

------------------------------------------------------------

PHASE 3: EXPLORATORY DATA ANALYSIS (Days 7-8)

DAY 7 – Statistical Analysis & Distribution Visualization

Heramb (Data Engineer)
- Generate comprehensive summary statistics
- Create data profiling reports
- Support Aayush with data preparation for visualizations

Aayush (Data Analyst) – LEAD
- Create histograms for all numerical features
- Visualize age distribution, income distribution, spending score distribution
- Generate box plots to identify outliers
- Document statistical insights

Shivam (ML Engineer)
- Analyze trends in customer spending behavior
- Identify potential cluster patterns from distributions
- Assess data separability for clustering

------------------------------------------------------------

DAY 8 – Relationship Analysis & Pattern Discovery

Heramb (Data Engineer)
- Prepare multi-dimensional datasets for visualization
- Create data aggregations for pattern analysis
- Generate correlation heatmaps

Aayush (Data Analyst) – LEAD
- Create scatter plots: Income vs Spending Score, Age vs Spending Score
- Generate pair plots for feature relationships
- Create correlation visualizations
- Document visual patterns and potential segments

Shivam (ML Engineer)
- Interpret visual patterns for clustering feasibility
- Identify natural groupings in scatter plots
- Estimate potential number of clusters from visual analysis

------------------------------------------------------------

PHASE 4: CLUSTERING IMPLEMENTATION (Days 9-12)

DAY 9 – Feature Engineering & Scaling

Heramb (Data Engineer) – LEAD
- Prepare final feature matrix for clustering
- Implement feature scaling (StandardScaler or MinMaxScaler)
- Create reusable preprocessing pipeline
- Save processed features to data/processed/

Aayush (Data Analyst)
- Validate scaled features and check distributions
- Document feature transformation decisions
- Create before/after scaling visualizations

Shivam (ML Engineer)
- Verify feature matrix format for Scikit-Learn
- Test preprocessing pipeline
- Prepare data splits if needed for validation

------------------------------------------------------------

DAY 10 – K-Means Implementation

Heramb (Data Engineer)
- Prepare feature matrix in optimal format
- Set up data structures for model training
- Create utility functions for model persistence

Aayush (Data Analyst)
- Prepare visualization scripts for cluster results
- Set up plotting functions for cluster analysis
- Create color schemes for cluster visualization

Shivam (ML Engineer) – LEAD
- Implement K-Means clustering using Scikit-Learn
- Create src/clustering.py with KMeans implementation
- Test clustering with initial k value (e.g., k=3)
- Document model parameters and configuration

------------------------------------------------------------

DAY 11 – Optimal Cluster Determination

Heramb (Data Engineer)
- Run clustering experiments with k ranging from 2 to 10
- Collect inertia values for each k
- Store results for analysis

Aayush (Data Analyst) – LEAD
- Generate Elbow Method graph (k vs inertia)
- Create silhouette score visualizations
- Document optimal k selection process
- Save plots to visuals/cluster_plots/

Shivam (ML Engineer)
- Analyze elbow curve and determine optimal cluster count
- Calculate silhouette scores for validation
- Finalize optimal k value based on multiple metrics
- Document cluster selection rationale

------------------------------------------------------------

DAY 12 – Cluster Visualization & Validation

Heramb (Data Engineer)
- Prepare dataset with cluster labels
- Create cluster summary statistics
- Generate cluster profile tables

Aayush (Data Analyst) – LEAD
- Create 2D scatter plots showing cluster distribution
- Generate 3D visualizations if using 3+ features
- Create cluster centroid visualizations
- Produce cluster size distribution charts
- Save all visualizations to visuals/cluster_plots/

Shivam (ML Engineer)
- Verify clustering quality and separation
- Calculate cluster evaluation metrics
- Analyze within-cluster and between-cluster variance
- Validate clustering results

------------------------------------------------------------

PHASE 5: INSIGHTS & RECOMMENDATIONS (Days 13-14)

DAY 13 – Cluster Interpretation & Profiling

Heramb (Data Engineer)
- Generate detailed cluster statistics (mean, median, std for each cluster)
- Create cluster profile reports
- Prepare data tables for business presentation

Aayush (Data Analyst)
- Describe characteristics of each customer group
- Create visual profiles for each segment
- Generate comparison charts between clusters
- Document behavioral patterns

Shivam (ML Engineer) – LEAD
- Analyze behavior patterns within each cluster
- Assign meaningful business names to clusters
- Create customer personas for each segment

Example Personas:
Cluster 0 – Budget Buyers (Low income, low spending, price-sensitive)
Cluster 1 – Premium Customers (High income, high spending, quality-focused)
Cluster 2 – Frequent Shoppers (Moderate income, high frequency, loyalty-driven)

------------------------------------------------------------

DAY 14 – Business Recommendations & Strategy

Heramb (Data Engineer)
- Prepare data-driven insights for each segment
- Calculate segment sizes and business impact
- Create actionable metrics for each segment

Aayush (Data Analyst)
- Explain targeting strategies with visual support
- Create segment comparison dashboards
- Document expected outcomes for each strategy

Shivam (ML Engineer) – LEAD
- Design marketing strategies for each customer segment
- Develop personalized recommendations per segment
- Prepare ROI projections and business case

Recommended Strategies:
Budget Buyers → Discount campaigns, clearance sales, bundle offers
Premium Customers → Exclusive launches, VIP programs, premium services
Frequent Shoppers → Loyalty rewards, subscription models, early access

------------------------------------------------------------

PHASE 6: DOCUMENTATION & DELIVERY (Days 15-16)

DAY 15 – Project Documentation

Heramb (Data Engineer) – LEAD
- Document data pipeline and preprocessing steps
- Write technical documentation for src/clustering.py
- Create data dictionary and schema documentation
- Document environment setup and dependencies

Aayush (Data Analyst) – LEAD
- Document all EDA insights and findings
- Create visualization guide explaining each chart
- Write analysis methodology documentation
- Compile all visual outputs with descriptions

Shivam (ML Engineer) – LEAD
- Write clustering methodology and algorithm explanation
- Document model parameters and hyperparameters
- Create results summary and performance metrics
- Write business recommendations document
- Update README.md with project outcomes

------------------------------------------------------------

DAY 16 – Final Review & Presentation Preparation

Heramb (Data Engineer)
- Prepare slides on dataset and data preprocessing
- Create data quality and cleaning summary
- Demonstrate data pipeline workflow

Aayush (Data Analyst)
- Prepare slides on visual analysis and insights
- Create compelling visualizations for presentation
- Demonstrate exploratory findings

Shivam (ML Engineer)
- Prepare slides on clustering algorithm and results
- Present business recommendations and strategies
- Create executive summary of project outcomes

Team Collaboration:
- Conduct final code review together
- Test complete workflow end-to-end
- Rehearse presentation as a team
- Prepare Q&A responses

------------------------------------------------------------

PROJECT WORKFLOW WITH OWNERSHIP

Customer Dataset
↓
[Heramb] Data Loading & Cleaning
↓
[Aayush] Exploratory Data Analysis & Visualization
↓
[Heramb] Feature Engineering & Scaling
↓
[Shivam] K-Means Clustering Implementation
↓
[Aayush] Cluster Visualization
↓
[Shivam] Customer Segmentation & Interpretation
↓
[Shivam] Business Recommendations

------------------------------------------------------------

DELIVERABLES BY TEAM MEMBER

Heramb (Data Engineer) Deliverables:
- Clean, processed dataset in data/processed/
- Data loading scripts in src/data_loader.py
- Data quality reports and documentation
- Feature engineering pipeline
- Technical documentation for data pipeline

Aayush (Data Analyst) Deliverables:
- Complete EDA notebook with insights
- All visualizations in visuals/cluster_plots/
- Statistical analysis reports
- Elbow method and silhouette score plots
- Cluster visualization dashboards
- Visual documentation and interpretation guide

Shivam (ML Engineer) Deliverables:
- K-Means implementation in src/clustering.py
- Optimal cluster determination analysis
- Cluster evaluation metrics
- Customer personas and segment profiles
- Business recommendations document
- Updated README.md with results
- Presentation slides on methodology and outcomes

------------------------------------------------------------

COLLABORATION CHECKPOINTS

End of Day 3: Team sync on environment setup and project structure
End of Day 6: Review cleaned data and feature selection together
End of Day 8: Review EDA insights and discuss clustering approach
End of Day 12: Review clustering results and validate segments
End of Day 14: Align on business recommendations
Day 16: Final presentation rehearsal

------------------------------------------------------------

SUCCESS CRITERIA

Technical Success:
- Clean dataset with no missing values or duplicates
- Clear cluster separation with optimal k value
- High silhouette score (>0.5)
- Reproducible clustering pipeline

Business Success:
- Actionable customer segments with clear characteristics
- Targeted marketing strategies for each segment
- Data-driven business recommendations
- Clear ROI potential for implementation

------------------------------------------------------------

FINAL PROJECT WORKFLOW

Customer Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Selection
↓
K-Means Clustering
↓
Customer Segmentation
↓
Business Recommendations