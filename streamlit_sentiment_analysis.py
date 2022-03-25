import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import zipfile
from zipfile import ZipFile

zf = zipfile.ZipFile('D:\LDS0\Week 3\sentiment_analysis.zip')

df_self = pd.read_csv(zf.open('data_Foody_HCM_Q13510.csv'))
df_self = df_self.drop(['Unnamed: 0', 'location', 'url'], axis=1)
df_self.columns = ['restaurant', 'review_text',	'review_score', 'avg_score']
df_self['review_text'] = df_self['review_text'].astype(str)
df_sub = df_self[['restaurant','review_text','review_score']]

df1 = pd.read_csv(zf.open('data_Foody.csv'))
df1 = df1.drop('Unnamed: 0', axis=1)

df = pd.concat([df_sub,df1],axis=0)
df = df.dropna()
df = df.drop_duplicates() 

df['reviewer_group'] = df['review_score'].apply(lambda x: '[1, 6)' if x < 6 else '6+') 
percentage_reviewer_group = df.groupby(by='reviewer_group')['review_score'].count()*100/df.shape[0]
percentage_reviewer_group.loc[['[1, 6)', '6+']].plot(kind='bar', color='darkblue', grid=True, figsize=(10, 7),title='Reviewer_Score_Group')

models_rec = pd.read_csv(zf.open('models3_recommendation_lazypredict.csv'))
models_rec = models_rec.sort_values(by=['Accuracy'], ascending=False)

models_6 = pd.read_csv(zf.open('models_6.csv'))
models_6 = models_6.sort_values(by=['Test Score'], ascending=False)

pkl_filename = "sentiment_classification_demo.pkl"
with open(pkl_filename, 'rb') as file:  
    tree = pickle.load(file)

pkl_tfidf = "tfidf_model.pkl" 
with open(pkl_tfidf, 'rb') as file:  
    tfidf_model = pickle.load(file)

st.title("Data Science Final Project")
st.write("## Restaurants Sentiment Analysis with Foody Ratings & Reviews")

import base64

# @st.cache(allow_output_mutation=True)
def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )

side_bg = 'background.png'
sidebar_bg(side_bg)

menu = ["Business Objective", 'EDA', 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox("Menu", menu)
if choice == 'Business Objective':
    st.subheader("I. Business Objective: ")
    st.write("""
    ###### Sentiment Analysis - phân tích tình cảm (hay còn gọi là phân tích quan điểm, phân tích cảm xúc, phân tính cảm tính, là cách sử dụng xử lý ngôn ngữ tự nhiên, phân tích văn bản, ngôn ngữ học tính toán, và sinh trắc học để nhận diện, trích xuất, định lượng và nghiên cứu các trạng thái tình cảm và thông tin chủ quan một cách có hệ thống. 
    """)
    st.write("""
    ###### Sentiment Analysis được áp dụng rộng rãi cho các tài liệu chẳng hạn như các đánh giá và các phản hồi khảo sát, phương tiện truyền thông xã hội, phương tiện truyền thông trực tuyến, và các tài liệu cho các ứng dụng từ marketing đến quản lý quan hệ khách hàng và y học lâm sàng.""")
    st.write("""
    ###### => Problem / Requirement: Xây dựng hệ thống hỗ trợ nhà hàng/quán ăn phân loại các phản hồi của khách hàng thành các nhóm: tích cực & tiêu cực dựa trên dữ liệu dạng văn bản là những comments từ Foody.vn.""")
    st.image('picture1.jpg')

elif choice == 'EDA':
    st.subheader("II. EDA:")
    st.image('eda.png')
    st.write("#### 1. Dataset Tự Scraping: ")
    st.write("Dataset tự cào có ", str(df_self.shape[0]), "dòng và", str(df_self.shape[1]), 'cột.')
    st.write("##### a. Data Display: ")
    st.dataframe(df_self.head(5))
    
    st.write("##### b. Visualize Some Data")
    fig1, ax1 = plt.subplots()
    ax1 = plt.subplot(2, 1, 1)
    ax1.hist(df_self['review_score'])
    ax1.set_title('Reviewer Score')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2 = plt.subplot(2, 1, 2)
    ax2.hist(df_self['avg_score'])
    ax2.set_title('Average Score')
    st.pyplot(fig2)
    st.write("###### Nhận xét: Nhìn chung, điểm ratings cá nhân & ratings trung bình của mỗi nhà hàng đều khá cao (Từ 6 điểm trở lên).")

    st.write("#### 2. Dataset có sẵn: ")
    st.write("Dataset có sẵn có ", str(df1.shape[0]), "dòng và", str(df1.shape[1]), 'cột.')
    st.write("##### a. Data Display: ")
    st.dataframe(df1.head(5))
    st.write("##### b. Visualize Some Data: ")
    fig3, ax3 = plt.subplots()
    ax3 = plt.subplot(1, 1, 1)
    ax3.hist(df1['review_score'])
    st.pyplot(fig3)
    st.write("###### Nhận xét: Tương tự dataset tự cào, điểm ratings cá nhân của mỗi nhà hàng cũng đều khá cao (Từ 6 điểm trở lên).")

    st.write("#### 3. Dataset kết hợp cả 2 dataset trên: ")
    st.write("Dataset kết hợp có ", str(df.shape[0]), "dòng và", str(df.shape[1]), 'cột.')
    st.write("##### a. Data Display: ")
    st.dataframe(df.head(5))
    st.write("##### b. Visualize Some Data: ")
    fig4, ax4 = plt.subplots()
    ax4 = plt.subplot(1, 1, 1)
    ax4.hist(df['review_score'])
    st.pyplot(fig4)
    st.write("###### Nhận xét #1: Với dataset này, điểm ratings cá nhân của mỗi nhà hàng cũng đều khá cao (Từ 6 điểm trở lên).")
    st.image('review_group.jpg')
    st.write("###### Nhận xét #2: Từ biểu đồ trên, hơn 80% ratings trên 6 điểm và dưới 20% ratings dưới 6 điểm.")
    st.write("###### Kết luận: Ta sẽ chia thành 2 nhóm sentiment để train model: Nhóm thứ nhất - Trên 6 điểm và  Nhóm thứ hai - Dưới 6 điểm. Tuy nhiên, vì có sự chênh lệch khá lớn giữa 2 nhóm, ta sẽ thực hiện Oversampling để train model.")

    st.write("#### 4. Oversampling: ")
    st.write("Sau khi Oversampling, Dataset có Dataset có 76604 dòng và 344 cột.")
    st.write("Mỗi class có số lượng tương đương nhau là 38302 dòng mỗi class.")

elif choice == 'Build Project': 
    st.subheader("III. Build Model:")
    st.image('models.png')
    st.write("##### 1. Lazy Predict Algorithm:")
    st.write("Thực hiện thuật toán Lazy Predict, ta có bảng thống kê như sau: ")
    st.dataframe(models_rec)
    st.write("###### Nhận xét #1: Từ bảng trên, dựa vào kết quả của LazyPredict, ta có thể cân nhắc sử dụng 1 trong 6 thuật toán chính là: Extra Tree Classifier, Extra Trees Classifier, Ridge Classifier, GaussianNB, Quadratic Discriminant Analysis và Logistic Regression.")
    st.write("##### 2. Training Models:")
    st.write("Thực hiện training models, ta có bảng thống kê accuracy / training time của 6 thuật toán này như sau: ")
    st.dataframe(models_6)
    st.write("###### Nhận xét #2: Từ bảng trên, ta có thể cân nhắc sử dụng ExtraTreeClassifier hoặc ExtraTreesClassifier vì 2 thuật toán này cho ra kết quả độ chính xác cao hơn so với các thuật toán còn lại. Tuy nhiên, nếu muốn tiết kiệm thời gian, ta nên sử dụng ExtraTreeClassifier (có thể thử tunning parameters).")
    
    st.write("##### 3. Tunning Parameters & Evaluation: ")
    
    st.write("##### 3.1. ExtraTreeClassifier: ")
    st.write("###### a. ROC Curve: ")
    st.image('picture2.jpg')
    st.write("###### b. Heatmap: ")
    st.image('picture3.jpg')
    st.write("###### Nhận xét #1: Sau khi tunning parameters với ExtraTreeClassifier, ta được ROC curve và Confusion Matrix như trên với accuracy_score vào khoảng 79.6%.")
    
    st.write("##### 3.2. ExtraTreesClassifier: ")
    st.write("###### a. ROC Curve: ")
    st.image('picture4.jpg')
    st.write("###### b. Heatmap: ")
    st.image('picture5.jpg')
    st.write("###### Nhận xét #2: Sau khi tunning parameters với ExtraTreesClassifier, ta được ROC curve và Confusion Matrix như trên với accuracy_score vào khoảng 84.2% (cao hơn so với thuật toán ExtraTreeClassifier). Tuy nhiên, thời gian training ExtraTreesClassifier lại lâu hơn và nặng hơn so với ExtraTreeClassifier nên ta có thể lưu ý 2 yếu tố này để chọn model phù hợp hơn khi thiết kế giao diện.")

    st.write("##### 4. Summary: ExtraTreeClassifier model tạm được để Sentiment Analysis Classification.")

elif choice == 'New Prediction':
    st.image('review.jpg')
    st.subheader("IV. New Prediction: ")
    flag = False
    lines = None
    type = st.radio('Upload or Input data', options=['Upload', 'Input'])
    if type == 'Upload':
        uploaded_file = st.file_uploader('Choose a file', type=['txt', 'csv'])
        if uploaded_file is not None:
            lines = pd.read_csv(uploaded_file, header=None)
            st.dataframe(lines)
            lines = lines[0]
            flag = True
    if type == 'Input':
        email = st.text_area(label = 'Input your content: ')
        if email != "":
            lines = np.array([email])
            flag = True

    if flag:
        st.write('Content: ')
        if len(lines) > 0:
            st.code(lines)
            x_new = tfidf_model.transform(lines)
            y_pred_new  = tree.predict(x_new)
            st.code('New prediction (0: Negative, 1: Positive): ' + str(y_pred_new))
            if y_pred_new == 0:
                st.write("It's so sad to know that you did not satisfy with the services.")
                st.image('sad_icon.png', width = 100)
            else:
                st.write("We're glad that you enjoyed the services.")
                st.image('happy_icon.jpg', width = 100)