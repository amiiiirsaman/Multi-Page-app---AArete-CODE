import streamlit as st
from PIL import Image
import cufflinks
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio 
import plotly.graph_objects as go
import warnings
import streamlit as st
warnings.filterwarnings("ignore")
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
pio.renderers.default = "notebook" # should change by looking into pio.renderers
from random import shuffle
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from numpy import where
from numpy import unique
from sklearn.mixture import GaussianMixture
from scipy import stats
import scipy.integrate
import scipy.special
import scipy
from PIL import Image
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import functions

img = Image.open("AArete.png")
st.set_page_config(layout = "wide", page_icon = 'AArete.png', page_title='CODE Demo', initial_sidebar_state="expanded", 
     menu_items={'About': "https://www.aarete.com/our-solutions/digital-technology/data-analytics/"})
               
st.markdown("<h1 style='text-align: left; color: #f68b28;'>CODE APP POC</h1>", unsafe_allow_html=True)
st.image(img, width=100, caption='Humanizing Data for Purpusful Change')

# Contents of ~/my_app/pages/page_2.py
def get_colors():
    s='''
        aliceblue, antiquewhite, aqua, aquamarine, azure,
        beige, bisque, black, blanchedalmond, blue,
        blueviolet, brown, burlywood, cadetblue,
        chartreuse, chocolate, coral, cornflowerblue,
        cornsilk, crimson, cyan, darkblue, darkcyan,
        darkgoldenrod, darkgray, darkgrey, darkgreen,
        darkkhaki, darkmagenta, darkolivegreen, darkorange,
        darkorchid, darkred, darksalmon, darkseagreen,
        darkslateblue, darkslategray, darkslategrey,
        darkturquoise, darkviolet, deeppink, deepskyblue,
        dimgray, dimgrey, dodgerblue, firebrick,
        floralwhite, forestgreen, fuchsia, gainsboro,
        ghostwhite, gold, goldenrod, gray, grey, green,
        greenyellow, honeydew, hotpink, indianred, indigo,
        ivory, khaki, lavender, lavenderblush, lawngreen,
        lemonchiffon, lightblue, lightcoral, lightcyan,
        lightgoldenrodyellow, lightgray, lightgrey,
        lightgreen, lightpink, lightsalmon, lightseagreen,
        lightskyblue, lightslategray, lightslategrey,
        lightsteelblue, lightyellow, lime, limegreen,
        linen, magenta, maroon, mediumaquamarine,
        mediumblue, mediumorchid, mediumpurple,
        mediumseagreen, mediumslateblue, mediumspringgreen,
        mediumturquoise, mediumvioletred, midnightblue,
        mintcream, mistyrose, moccasin, navajowhite, navy,
        oldlace, olive, olivedrab, orange, orangered,
        orchid, palegoldenrod, palegreen, paleturquoise,
        palevioletred, papayawhip, peachpuff, peru, pink,
        plum, powderblue, purple, red, rosybrown,
        royalblue, saddlebrown, salmon, sandybrown,
        seagreen, seashell, sienna, silver, skyblue,
        slateblue, slategray, slategrey, snow, springgreen,
        steelblue, tan, teal, thistle, tomato, turquoise,
        violet, wheat, white, whitesmoke, yellow,
        yellowgreen
        '''
    li=s.split(',')
    li=[l.replace('\n','') for l in li]
    li=[l.replace(' ','') for l in li]
    shuffle(li)
    return li

colors = get_colors()   
data = pd.read_csv("testckd2.csv")
data = data.drop(columns=['MEMBER_ID_UNIVERSAL'])
                 
data.describe()

columns = ['Diabetes', 'Behavioral Health/Substance Abuse History',
       'Average_Adherance', 'Inpatient_CKD_Visits_by_Month',
       'Outpatient_Visits_by_Month', 'Comorbidity']

def main_page():
    st.markdown("# Main page ")
    st.sidebar.markdown("# Main page")
    st.markdown('***')
    st.markdown("##### This is the main page of the tool developed for Exploratory data analysis (EDA) and Clustering Modeling by AArete Data Excellence Team! Please check out the next pages and feel free to reach out to us if you have any questions or concerns!")

def page2():
    st.markdown("# Exploratory Data Analysis")
    st.sidebar.markdown("# Exploratory Data Analysis")

    mode = "EDA"

    if mode=="EDA":
        analysis_type = st.sidebar.radio("Analysis Type", ["Profiling Report", "Detailed Version"])
        st.markdown(f"# Analysis Mode: {analysis_type}")

        functions.space()
        st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)
        df = data
        file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
        dataset = st.file_uploader(label = '')

        use_defo = st.checkbox('Use example Dataset')
        if use_defo:
            dataset = 'testckd2.csv'

        if dataset:
            if file_format == 'csv' or use_defo:
                df = pd.read_csv(dataset)
            else:
                df = pd.read_excel(dataset)    


        if analysis_type=="Profiling Report":
            columns = df.columns.to_list()
            featuress = st.sidebar.multiselect("Select Locations ", columns, default=columns[:3])

            df = df[featuress]
            pr = ProfileReport(df, explorative=True)
            st.header('**Input DataFrame**')
            st.write(df)                  
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)

        if analysis_type=="Detailed Version":
            dff = df
            st.subheader('Dataframe:')
            n, m = data.shape
            st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
            st.dataframe(dff)

            all_vizuals = ['Dataframe Information', 'Missingness Evaluation', 'Descriptive Analysis', 'Target Variable Analysis', 
                           'Distribution of Numerical Features', 'Count Plots of Categorical Features', 
                           'Box Plots of Numerical Features', 'Outlier Analysis', 'Variance of Target with Categorical Features']
            functions.sidebar_space(3)         
            vizuals = st.sidebar.multiselect("Choose which visualizations you want to see:", all_vizuals)

            if 'Dataframe Information' in vizuals:
                st.subheader('Info:')
                c1, c2, c3 = st.columns([1, 2, 1])
                c2.dataframe(functions.df_info(dff))

            if 'Missingness Evaluation' in vizuals:
                st.subheader('Missingness Evaluation:')
                if dff.isnull().sum().sum() == 0:
                    st.write('There is not any NA value in your dataset.')
                else:
                    c1, c2, c3 = st.columns([0.5, 2, 0.5])
                    c2.dataframe(functions.df_isnull(dff), width=1500)
                    functions.space(2)


            if 'Descriptive Analysis' in vizuals:
                st.subheader('Descriptive Analysis:')
                st.dataframe(dff.describe())

            if 'Target Variable Analysis' in vizuals:
                st.subheader("Select target column:")    
                target_column = st.selectbox("", dff.columns, index = len(dff.columns) - 1)

                st.subheader("Histogram of target column")
                fig = px.histogram(dff, x = target_column)
                c1, c2, c3 = st.columns([0.5, 2, 0.5])
                c2.plotly_chart(fig)


            num_columns = dff.select_dtypes(exclude = 'object').columns
            cat_columns = dff.select_dtypes(include = 'object').columns

            if 'Distribution of Numerical Features' in vizuals:

                if len(num_columns) == 0:
                    st.write('There is no numerical columns in the data.')
                else:
                    selected_num_cols = functions.sidebar_multiselect_container('Choose columns for Distribution plots:', num_columns, 'Distribution')
                    st.subheader('Distribution of Numerical Features')
                    i = 0
                    while (i < len(selected_num_cols)):
                        c1, c2 = st.columns(2)
                        for j in [c1, c2]:

                            if (i >= len(selected_num_cols)):
                                break

                            fig = px.histogram(dff, x = selected_num_cols[i])
                            j.plotly_chart(fig, use_container_width = True)
                            i += 1

            if 'Count Plots of Categorical Features' in vizuals:

                if len(cat_columns) == 0:
                    st.write('There is no categorical columns in the dff.')
                else:
                    selected_cat_cols = functions.sidebar_multiselect_container('Choose columns for Count plots:', cat_columns, 'Count')
                    st.subheader('Count Plots of Categorical Features')
                    i = 0
                    while (i < len(selected_cat_cols)):
                        c1, c2 = st.columns(2)
                        for j in [c1, c2]:

                            if (i >= len(selected_cat_cols)):
                                break

                            fig = px.histogram(dff, x = selected_cat_cols[i], color_discrete_sequence=['indianred'])
                            j.plotly_chart(fig)
                            i += 1

            if 'Box Plots of Numerical Features' in vizuals:
                if len(num_columns) == 0:
                    st.write('There is no numerical columns in the dff.')
                else:
                    selected_num_cols = functions.sidebar_multiselect_container('Choose columns for Box plots:', num_columns, 'Box')
                    st.subheader('Box Plots of Numerical Features')
                    i = 0
                    while (i < len(selected_num_cols)):
                        c1, c2 = st.columns(2)
                        for j in [c1, c2]:

                            if (i >= len(selected_num_cols)):
                                break

                            fig = px.box(dff, y = selected_num_cols[i])
                            j.plotly_chart(fig, use_container_width = True)
                            i += 1

            if 'Outlier Analysis' in vizuals:
                st.subheader('Outlier Analysis')
                c1, c2, c3 = st.columns([1, 2, 1])
                c2.dataframe(functions.number_of_outliers(dff))

            if 'Variance of Target with Categorical Features' in vizuals:


                df_1 = dff.dropna()

                high_cardi_columns = []
                normal_cardi_columns = []

                for i in cat_columns:
                    if (dff[i].nunique() > dff.shape[0] / 10):
                        high_cardi_columns.append(i)
                    else:
                        normal_cardi_columns.append(i)


                if len(normal_cardi_columns) == 0:
                    st.write('There is no categorical columns with normal cardinality in the data.')
                else:

                    st.subheader('Variance of target variable with categorical Features')
                    model_type = st.radio('Select Problem Type:', ('Regression', 'Classification'), key = 'model_type')
                    selected_cat_cols = functions.sidebar_multiselect_container('Choose columns for Category Colored plots:', normal_cardi_columns, 'Category')

                    if 'Target Variable Analysis' not in vizuals:   
                        target_column = st.selectbox("Select target column:", dff.columns, index = len(dff.columns) - 1)

                    i = 0
                    while (i < len(selected_cat_cols)):

                        if model_type == 'Regression':
                            fig = px.box(df_1, y = target_column, color = selected_cat_cols[i])
                        else:
                            fig = px.histogram(df_1, color = selected_cat_cols[i], x = target_column)

                        st.plotly_chart(fig, use_container_width = True)
                        i += 1

                    if high_cardi_columns:
                        if len(high_cardi_columns) == 1:
                            st.subheader('The following column has high cardinality, that is why its boxplot was not plotted:')
                        else:
                            st.subheader('The following columns have high cardinality, that is why its boxplot was not plotted:')
                        for i in high_cardi_columns:
                            st.write(i)

                        st.write('<p style="font-size:140%">Do you want to plot anyway?</p>', unsafe_allow_html=True)    
                        answer = st.selectbox("", ('No', 'Yes'))

                        if answer == 'Yes':
                            for i in high_cardi_columns:
                                fig = px.box(df_1, y = target_column, color = i)
                                st.plotly_chart(fig, use_container_width = True)    

def page3():
    st.markdown("# Clustering Modeling")
    st.sidebar.markdown("# Clustering Modeling")
    
    mode="Clustering"

    if mode=="Clustering":    
        features = st.sidebar.multiselect("Select Features", columns, default=columns)

        # select a clustering algorithm
        calg = st.sidebar.selectbox("Select a clustering algorithm", ["K-Means", "Gaussian Mixture Clustering"])

        # select number of clusters
        ks = st.sidebar.slider("Select number of clusters", min_value=2, max_value=7, value=2)

        # select a dataframe to apply cluster on


        if len(features)>=2:

            if calg == "K-Means":
                st.markdown("### K-Means Clustering")        
                use_pca = st.sidebar.radio("Use PCA?",["Yes","No"])
                if use_pca=="No":
                    st.markdown("### Not Using PCA")
                    inertias = []
                    for c in range(2,ks+1):
                        X = data[features]                
                        # colors=['red','green','blue','magenta','black','yellow']
                        model = KMeans(n_clusters=c, n_init=10, random_state = 29)
                        model.fit(X)
                        y_kmeans = model.predict(X)
                        centers = model.cluster_centers_
                        clusters_df = pd.DataFrame(centers, columns=features)
                        cluster_names = [f'Cluster {k}' for k in range(1,c+1, 1)]  # clust 1, 2, 3
                        dataa = [go.Bar(name=f, x=cluster_names, y=clusters_df[f]) for f in clusters_df.columns]
                        fig = go.Figure(data=dataa)
                        # Change the bar mode
                        fig.update_layout(barmode='group')
                        st.plotly_chart(fig)

                if use_pca=="Yes":
                    st.markdown("### Using PCA")
                    comp = st.sidebar.number_input("Choose Components",1,10,3)

                    X = data[features]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    pca = PCA(n_components=int(comp))
                    principalComponents = pca.fit_transform(X_scaled)
                    feat = list(range(pca.n_components_))
                    PCA_components = pd.DataFrame(principalComponents, columns=list(range(len(feat))))
                    chosen_component = st.sidebar.multiselect("Choose Components",feat,default=[1,2])
                    chosen_component=[int(i) for i in chosen_component]
                    inertias = []
                    if len(chosen_component)>1:
                        for c in range(2,ks+1):
                            X = PCA_components[chosen_component]

                            model = KMeans(n_clusters=c)
                            model.fit(X)

                            trace0 = go.Scatter(x=X[chosen_component[0]],y=X[chosen_component[1]],mode='markers',  
                                                marker=dict(
                                                        color=colors,
                                                        colorscale='Viridis',
                                                        showscale=True,
                                                        opacity = 0.9,
                                                        reversescale = True,
                                                        symbol = 'pentagon'
                                                        ))

                            trace1 = go.Scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1],
                                                mode='markers', 
                                                marker=dict(
                                                    color=colors,
                                                    size=20,
                                                    symbol="circle",
                                                    showscale=True,
                                                    line = dict(
                                                        width=1,
                                                        color='rgba(102, 102, 102)'
                                                        )

                                                    ),
                                                name="Cluster Mean")

                            data7 = go.Data([trace0, trace1])
                            fig = go.Figure(data=data7)

                            layout = go.Layout(
                                        height=600, width=800, title=f"KMeans Cluster Size {c}",
                                        xaxis=dict(
                                            title=f"Component {chosen_component[0]}",
                                        ),
                                        yaxis=dict(
                                            title=f"Component {chosen_component[1]}"
                                        ) ) 
                            fig.update_layout(layout)
                            st.plotly_chart(fig)                                 


            elif calg == "Gaussian Mixture Clustering":
                st.markdown("### GMM Clustering")        
                use_pca = st.sidebar.radio("Use PCA?",["Yes","No"])
                if use_pca=="No":
                    st.markdown("### Not Using PCA")
                    inertias = []
                    for c in range(2,ks+1):
                        X = data[features]                
                        # colors=['red','green','blue','magenta','black','yellow']
                        Y = X.to_numpy()
                        gmm = GaussianMixture(n_components=c, covariance_type='full').fit(Y)
                        centers = np.empty(shape=(gmm.n_components, Y.shape[1]))
                        for i in range(gmm.n_components):
                            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(Y)
                            centers[i, :] = Y[np.argmax(density)]
                        clusters_df = pd.DataFrame(centers, columns=features)
                        cluster_names = [f'Cluster {k}' for k in range(1,c+1, 1)]  # clust 1, 2, 3
                        dataa = [go.Bar(name=f, x=cluster_names, y=clusters_df[f]) for f in clusters_df.columns]
                        fig = go.Figure(data=dataa)
                        # Change the bar mode
                        fig.update_layout(barmode='group')
                        st.plotly_chart(fig)
                    
def page4():
    st.markdown("# Get In Touch")
    st.sidebar.markdown("# Get In Touch")

    st.markdown('***')
    st.markdown('Author: CODE Team at AArete LLC')
    st.markdown(
        "Thanks for using this tool! If you want to reach out you can find us on [LinkedIn] (https://www.linkedin.com/company/aarete/mycompany/) or our [website](https://www.aarete.com/).")    
    
    
page_names_to_funcs = {
    "Main Page": main_page,
    "Exploratory Data Analysis": page2,
    "Clustering Modeling": page3,
    "Get In Touch": page4,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()




                









