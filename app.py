import streamlit as st 
import altair as alt
from sklearn.datasets import load_iris
import pandas as pd
import plotly.express as px

st.subheader('Apolonio Manuel')

st.title('Practice Streamlit, Iris Dataset')

st.markdown("""
    In this example, we will use the classic Iris dataset.
    will explore the data and train a simple model.
""")

# Load the iris dataset
iris = load_iris()

# Create a DataFrame with feature names
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target column to the DataFrame
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

st.title('Iris Dataset')
st.dataframe(iris_df)

# Estadisticas descriptivas
st.title('Descriptive Statistics')
st.write(iris_df.select_dtypes(include='float').describe())


st.title('Scatter Plot')
# Create a scatter plot
scatter_chart = alt.Chart(iris_df).mark_circle().encode(
    x='petal length (cm)',
    y='petal width (cm)',
    color='species'
).interactive()

st.altair_chart(scatter_chart, use_container_width=True)

st.title('3D Scatter Plot')

chart_3d = px.scatter_3d(iris_df, 
                         x='sepal length (cm)', 
                         y='sepal width (cm)', 
                         z='petal length (cm)', 
                         color='species',
                         size='petal width (cm)',
                         size_max= 15
                         )

st.plotly_chart(chart_3d)

