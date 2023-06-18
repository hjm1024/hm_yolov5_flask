import pandas as pd
from sklearn.cluster import KMeans

# (1)从数据库中读取竞赛网站的用户访问数据。
data = pd.read_csv('user_data.csv', encoding='gbk')

# (2)对数据进行数据清洗、网页分类，得到网页分类的数据。
def classify_url(url):
    if 'www.tipdm.com/type1' in url:
        return 'Type 1'
    elif 'www.example.com/type2' in url:
        return 'Type 2'
    else:
        return 'Other'
# apply函数对整个数据集进行分类
data['Page Type'] = data['URL'].apply(classify_url)

# (3)使用预处理后的数据进行构造特征，构建用户访问不同类别网页数量的特征。
feature_matrix = data.groupby(['User', 'Page Type']).size().unstack(fill_value=0)

# (4)使用K-Means聚类算法根据构造的特征对用户进行分群。
# 设定聚类的个数
n_clusters = 5
# 创建KMeans聚类模型并进行训练
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(feature_matrix)

# (5)对不同的分类人群进行分析并提出建议。
