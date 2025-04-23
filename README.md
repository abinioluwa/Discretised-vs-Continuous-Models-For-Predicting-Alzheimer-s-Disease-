# Discretised-vs-Continuous-Models-For-Predicting-Alzheimer-s-Disease-
This study is an exploration of Feature Correlation Approaches in Machine Learning Based Solutions for the Diagnosis of Alzheimer’s Disease

This study employs a comprehensive machine learning approach to classify and analyse (AD) stages, with a particular focus on comparing the effects of discretising continuous features versus retaining them as continuous variables during the feature correlation analysis stage. The methodology is inspired by and builds upon the work of Alatrany et al. (2024), who developed an explainable machine learning approach for AD classification. The methodology consists of five main stages:
1.	Data Acquisition
2.	Data Preprocessing
3.	Feature Selection
4.	Train-Test Split
5.	Machine Learning Model Building,

Each stage is designed to address specific challenges in AD classification and to provide insights into the impact of feature discretisation on model performance and interoperability.

## Data Acquisition
The primary dataset used in this study is obtained from the National Alzheimer's Coordinating Center (NACC) database. This comprehensive dataset is a categorical and numerical dataset which contains a wide range of clinical, neuropsychological, and demographic information from participants across various cognitive states, including normal cognition (NC), mild cognitive impairment (MCI), and Alzheimer's disease (AD).
### Legal & Ethical Considerations
The utilisation of data from the NACC database in this study necessitates careful attention to legal and ethical considerations. The NACC data is freely available to researchers upon submission of a Quick-Access file request detailing the research purposes for which the data is needed. Upon a careful review of the request by the NACC, the data is made available to researchers whose proposals are within ethical and legal rights (NACC, 2024). However, this accessibility comes with a commitment to using the data solely for the stated research purposes and in accordance with all applicable data protection regulations.

It is crucial to note that all participants and co-participants in the NACC database participated with written and informed consent for their data to be used in research (Beekly et al., 2007). This consent forms the ethical foundation for this study.

As noted by Pang et al. (2023), NACC database participants do not constitute a statistically representative sample of the U.S. population. The data is best regarded as a referral-based or volunteer case series, which impacts the generalisability of any research findings. Hence, the study cannot result in any broad population-level inferences.

Given that the NACC dataset includes data from individuals with varying degrees of cognitive impairment, it is important that all analysis and reporting are conducted with sensitivity, ensuring that the findings do not stigmatise or misrepresent individuals with cognitive impairments while maintaining objectivity. In lieu of this the NACC’s data is anonymous and cannot be used to identify or infer any participants (NACC, 2024).

These considerations guide every aspect of the research method, from data handling to the interpretation and dissemination of results.

## Data Preprocessing
### Feature Selection Based on Literature
Following the approach of Alatrany et al. (2024), we begin by removing features from the NACC dataset based on a thorough literature review. This step is crucial for focusing on the most relevant variables associated with AD progression and diagnosis. 

The selection criteria prioritise features that have demonstrated significant correlations with AD in previous studies, including:
•	Demographic factors: age, gender, education level
•	Cognitive assessment scores: Mini-Mental State Examination (MMSE), Clinical Dementia Rating (CDR)
•	Neuropsychological test results: Trail Making Test, Logical Memory Test
•	Genetic markers: APOE ε4 allele status
•	Neuroimaging measures: hippocampal volume, cortical thickness in AD-sensitive regions

### Dataset Splitting by Class Label
To facilitate multi-class and binary classification tasks, the preprocessed dataset is partitioned into four subsets:
1.	NC vs. AD: For binary classification between normal cognition and Alzheimer's disease
2.	NC vs. MCI: For distinguishing between normal cognition and mild cognitive impairment
3.	MCI vs. AD: For differentiating between mild cognitive impairment and Alzheimer's disease
4.	NC vs. MCI vs. AD: For multi-class classification across all three cognitive states

This stratified approach allows for targeted analysis of the transition between cognitive states and enables the development of specialised models for each classification task.

### Data Transformation
To enhance the quality and consistency of the data, several transformation techniques to the dataset before applying any ML techniques:
#### 1.	Handling Missing/ Null
In categorical (object) columns, missing or null values are replaced with an "empty" category rather than being imputed or removed. This approach is based on the understanding that in medical datasets, the absence of information often carries significant meaning (Hou et al., 2022). For instance, a missing value in a symptom-related field might indicate the absence of that symptom, which is clinically relevant information. Numerical columns also contained specific values (e.g., 888.8, 88.8, 8.8, 9, 99, 999, 88, 888, -4, and 8888) that represent unknown or not applicable data in the context of the NACC dataset (NACC,2024). These values are uniformly replaced with NaN (Not a Number) to standardise the representation of missing numerical data. This step ensures consistency in how missing data is handled across all numerical features.

Following the standardisation of missing values, imputation is performed to address the remaining NaN values in numerical features. For continuous features, missing values are replaced with the mean of the respective feature, while for discrete features, the mode is used. This approach, while simple, has been shown to be effective in preserving the overall distribution of the data while minimising bias.

#### 2.	Removal of Low-Variance Features
Features that lack variability, defined as those where 90% or more of the values are identical, are removed from the dataset. This step is crucial for dimensionality reduction and improving model efficiency, as low-variance features typically contribute little to the predictive power of a model and can sometimes introduce noise (Liu et al., 2014).

#### 3.	Outlier Detection and Treatment
Outliers are identified using the interquartile range (IQR) method, where values falling below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR are considered outliers. These outliers are then replaced with the median value of the respective feature. This approach helps to mitigate the impact of extreme values on model performance while preserving the overall data distribution.

#### 4.	Removal of Redundant Categorical Variables
Categorical variables deemed redundant based on domain knowledge and statistical analysis are removed. This step helps to reduce multicollinearity (i.e., when two or more independent variables are highly correlated) and simplify the model without significant loss of information (Zhang et al., 2021).

#### 5.	Discretization of Continuous Features
In the first stage of the experiment, continuous features are discretised. This process involves converting continuous numerical values into discrete categories. Discretisation can help in revealing non-linear relationships and can make the data more robust to outliers and small fluctuations. The equal-frequency binning method is employed, dividing the range of a continuous variable into a specified number of bins, each containing approximately the same number of samples.

## Feature Selection
The feature selection process is conducted in two phases, employing different approaches for discretised and continuous data:

For the first phase, working with discretised values, the Mutual Information (MI) approach is utilised. MI measures the mutual dependence between two variables and is particularly useful for capturing non-linear relationships (Liu et al., 2002). It quantifies the amount of information obtained about one variable by observing the other variable. Features are ranked based on their MI scores with the target variable, and a subset of top-ranking features is selected.

In the second phase, which retains continuous values, Spearman's rank correlation approach is employed. This non-parametric measure assesses the monotonicity of the relationship between two variables without making assumptions about the distribution of the data. Features are selected based on their correlation strength with the target variable, with a threshold typically set to exclude weak correlations.

The two-stage approach to feature selection allows for a comprehensive evaluation of feature importance across different data representations. By comparing the features selected in each stage, insights can be gained into the impact of discretisation on feature relevance and the robustness of feature importance across different analytical approaches.

## Train-Test Split
The train-test split employs a two-stage splitting approach, aligning with best practices in machine learning for healthcare applications (Lundberg et al., 2020). Initially, the NCvsAD dataset is divided into a 70% training set and a 30% temporary set using sci-kit-learn's train_test_split function, with stratification based on the 'NACCUDSD' variable to maintain class distribution (Pedregosa et al., 2021). The temporary set is then equally split into testing and validation sets, each representing 15% of the original dataset, again utilising stratification. This approach ensures clear separation between data used for model training, hyperparameter tuning, and final evaluation, minimising the risk of information leakage. The resulting partitions—70% training, 15% testing, and 15% validation—balance the need for a substantial training set with rigorous model validation and testing, adhering to recent recommendations in biomedical machine learning.

## Machine Learning Model Building
The machine learning model building stage involves the development and evaluation of multiple classifiers to address the various classification tasks defined earlier. We implement four widely recognised machine learning algorithms, each chosen for its unique strengths and interpretability characteristics.

### Algorithm Selection
The following algorithms are implemented:
1.	Random Forest (RF): Random Forest is an ensemble learning method that constructs multiple decision trees and merges them to get a more accurate and stable prediction. It is chosen for its ability to handle high-dimensional data, capture complex interactions, and provide feature importance measures.
2.	K-Nearest Neighbors (KNN): KNN is a non-parametric method used for classification and regression. Its simplicity and effectiveness in capturing local patterns make it a valuable inclusion in our model suite.
3.	Naïve Bayes (NB): Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features. It is included for its computational efficiency and effectiveness in high-dimensional settings.
4.	Support Vector Machine (SVM): SVM is a powerful algorithm that finds the hyperplane that best divides a dataset into classes. It is chosen for its effectiveness in high-dimensional spaces and versatility through the use of different kernel functions.

### Model Training
Each algorithm is trained on the preprocessed and feature-selected training data for both the discretised and continuous feature versions. We use stratified k-fold cross-validation (k=5) to ensure robust performance estimation and to mitigate overfitting.

### Hyperparameter Optimisation
To optimise the performance of each model, we employ grid search with cross-validation for hyperparameter tuning. The hyperparameters tuned for each algorithm are:
•	Random Forest: number of trees, maximum depth, minimum samples split
•	KNN: number of neighbors, weight function
•	Naive Bayes: smoothing parameter (alpha) for Gaussian NB
•	SVM: kernel type, C parameter, gamma

### Model Evaluation
The performance of each model is evaluated using the following metrics:
•	Accuracy: Overall correctness of the model
•	Precision: The ability of the model to avoid labelling negative samples as positive
•	Recall: Ability of the model to find all positive samples
•	F1-score: Harmonic mean of precision and recall
•	Area Under the Receiver Operating Characteristic curve (AUC-ROC): Model's ability to distinguish between classes

These metrics are calculated for each fold of the cross-validation and averaged to provide robust performance estimates.

### Comparison of Discretised vs. Continuous Models
We conduct a comparative analysis of the models trained on discretised features versus those trained on continuous features. This comparison includes:
•	Performance metric differences
•	Stability of results across cross-validation folds
•	Computational efficiency
•	Interpretability of model decisions

### External Validation
To assess the generalisability of our models, external validation was performed using an independent validation set obtained during the train-test split. This step ensures that our models' performance is not overly optimistic due to potential biases in the NACC dataset.

### Comparison of Rules from Discretised vs. Continuous Models
We conduct a comparative analysis of the rules extracted from models trained on discretised features versus those trained on continuous features. This comparison focuses on differences in rule complexity and interpretability, variations in feature importance as reflected in the rules and clinical insights unique to each approach

### Integration of Findings
The insights gained from the rule extraction process are integrated with the performance metrics and feature importance rankings from earlier stages. This holistic analysis aims to provide a comprehensive understanding of:
- The most robust predictors of AD across different modelling approaches
- The impact of feature discretisation on model interpretability and clinical insights
- Potential new hypotheses about AD progression and risk factors

This methodology presents a systematic approach to AD classification and analysis, with a novel focus on comparing the effects of feature discretisation. By implementing a multi-stage process that encompasses preprocessing, feature selection, model building, and rule extraction, we aim to provide both accurate classification models and interpretable insights into AD progression. The comparison between discretised and continuous feature approaches throughout each stage offers valuable perspectives on the trade-offs between model performance, interpretability, and clinical relevance in the context of AD research.

The integration of advanced machine learning techniques with rigorous statistical analysis and domain-specific knowledge allows for a comprehensive exploration of AD classification challenges. Furthermore, the emphasis on model interpretability through rule extraction techniques addresses the critical need for explainable AI in healthcare applications.

This methodology builds upon and extends the work of Alatrany et al. (2024), offering novel contributions in the areas of feature discretisation analysis and the application of state-of-the-art rule extraction techniques. The insights gained from this study have the potential to inform clinical decision-making processes and contribute to the broader understanding of Alzheimer's disease progression and risk factors.

#References
ALATRANY, A.S., KHAN, W., HUSSAIN, A., KOLIVAND, H., and AL-JUMEILY, D., 2024. An explainable machine learning approach for Alzheimer’s disease classification. Scientific Reports [online]. Available from: http://dx.doi.org/10.1038/s41598-024-51985-w [Accessed 2 Apr. 2024]

BEEKLY, D.L., RAMOS, E.M., LEE, W.W., DEITRICH, W.D., JACKA, M.E., WU, J., HUBBARD, J.L., KOEPSELL, T.D., MORRIS, J.C. and KUKULL, W.A. (2007). The National Alzheimer’s Coordinating Center (NACC) Database: The Uniform Data Set. Alzheimer Disease & Associated Disorders, [online] 21(3), pp.249–258. doi: 10.1097/WAD.0b013e318142774e.

HOU, J., ZHAO, R., GRONSBELL, J., BEAULIEU-JONES, B.K., WEBBER, G., JEMIELITA, T., WAN, S., HONG, C., LIN, Y., CAI, T., WEN, J., PANICKAN, V.A., BONZEL, C.-L., LIAW, K.-L., LIAO, K.P., and CAI, T., 2022. Harnessing electronic health records for real-world evidence. [online]. Available from: https://arxiv.org/abs/2211.16609

LIU, S., et al. (2014). Multimodal neuroimaging feature learning for multiclass diagnosis of Alzheimer's disease. IEEE Transactions on Biomedical Engineering, 62(4), 1132-1140.

LUNDBERG, S.M., ERION, G., CHEN, H., DEGRAVE, A., PRUTKIN, J.M., NAIR, B., KATZ, R., HIMMELFARB, J., BANSAL, N., and LEE, S.-I., 2020. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence [online]. Available from: http://dx.doi.org/10.1038/s42256-019-0138-9.

NATIONAL ALZHEIMER'S COORDINATING CENTER (NACC) (2024). About NACC Data | National Alzheimer’s Coordinating Center. [online] naccdata.org. Available at: https://naccdata.org/requesting-data/nacc-data

PANG, Y., KUKULL, W., SANO, M., ALBIN, R.L., SHEN, C., ZHOU, J. and DODGE, H.H. (2023). Predicting Progression from Normal to MCI and from MCI to AD Using Clinical Variables in the National Alzheimer’s Coordinating Center Uniform Data Set Version 3: Application of Machine Learning Models and a Probability Calculator. The Journal of Prevention of Alzheimer’s Disease, 10. doi: 10.14283/jpad.2023.10.

PEDREGOSA, F., VAROQUAUX, G., GRAMFORT, A., MICHEL, V., THIRION, B., GRISEL, O., BLONDEL, M., MÜLLER, A., NOTHMAN, J., LOUPPE, G., PRETTENHOFER, P., WEISS, R., DUBOURG, V., VANDERPLAS, J., PASSOS, A., COURNAPEAU, D., BRUCHER, M., PERROT, M., and DUCHESNAY, É., 2012. Scikit-learn: Machine Learning in Python. arXiv [online]. Available from: https://arxiv.org/abs/1201.0490.

ZHANG, Y., ZHU, R., CHEN, Z., GAO, J., and XIA, D., 2021. Evaluating and selecting features via information theoretic lower bounds of feature inner correlations for high-dimensional data. European Journal of Operational Research [online]. Available from: http://dx.doi.org/10.1016/j.ejor.2020.09.028.
