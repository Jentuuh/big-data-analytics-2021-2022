# Big Data Analytics Course Assignments

# Summary 
* *Frequent Itemsets* : Trying to find groups of authors that frequently publish together. The algorithms that we implemented are A-priori, PCY and optimizations of PCY (PCY hashing, multi-hashing).
* *Clustering* : Clustered scientific papers into several subjects per time period to analyze the evolution and scientific progression within that certain field. The final algorithm that we used was a modified version of k-means clustering (`k_means_custom.py`). We estimated the right amount of clusters using the elbow method.
* *Data Management* : Detected similar documents within a large set of documents using Local Sensitive Hashing (LSH). The implementation was made in Spark (pyspark). For performance comparison a brute-force implementation as well as our LSH implementation `lsh_spark_alt.py` was made. 

Hasselt University, 2021-2022
