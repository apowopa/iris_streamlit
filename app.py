import streamlit as st 
import altair as alt
from sklearn.datasets import load_iris
import pandas as pd
import plotly.express as px
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


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

st.subheader('Iris Dataset')
st.dataframe(iris_df)

# Estadisticas descriptivas
st.subheader('Descriptive Statistics')
st.write(iris_df.select_dtypes(include='float').describe())


st.subheader('2D Scatter Plot')
# Create a scatter plot
scatter_chart = alt.Chart(iris_df).mark_circle().encode(
    x='petal length (cm)',
    y='petal width (cm)',
    color='species'
).interactive()

st.altair_chart(scatter_chart, use_container_width=True)

st.subheader('3D Scatter Plot')

chart_3d = px.scatter_3d(iris_df, 
                         x='sepal length (cm)', 
                         y='sepal width (cm)', 
                         z='petal length (cm)', 
                         color='species',
                         size='petal width (cm)',
                         size_max= 15
                         )

st.plotly_chart(chart_3d)

st.subheader('Train a simple model')

X = iris_df.drop(['target', 'species'], axis=1)
y = iris_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = SVC()

model.fit(X_train, y_train)

# el usuario puede seleccionar un registro para que el modelo haga una predicción

st.markdown('this model was a Support Vector Classification [SVC] model')

st.subheader('Make a prediction')

# seleccionar un registro de test de forma interactiva

index = st.selectbox('Select a test sample by the index', range(len(X_test)) , index=None)

if index is not None:
    # mostrar los datos del registro seleccionado
    selected_data = X_test.iloc[index]
    st.write(selected_data)
    st.write(f'The target is: {iris.target_names[y_test.iloc[index]]}')

    # hacer una predicción
    prediction = model.predict([selected_data])
    st.write(f'The prediction is: {iris.target_names[prediction][0]}')

    st.subheader('Model Evaluation')
    score = model.score(X_test, y_test)
    st.write(f'The model accuracy is: {score:.2f}')

    st.write('i mean is the iris dataset, the model is very simple, but it is a good example to show how to use Streamlit')



