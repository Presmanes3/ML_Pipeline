import streamlit as st

st.set_page_config(page_title="Problem Analysis", layout="wide")

st.title("Problem Analysis: Market Insights")

st.markdown("Este proyecto utiliza un dataset de propiedades en venta de NY, que puede ser encontrado en [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market)")


with st.expander("Price over location", expanded=False):
    st.markdown("""
                En primer lugar, hemos superpuesto los precios de las propiedades en venta sobre un mapa de la ciudad de nueva york, para ver una distribucion geográfica de los precios.
                
                """)

    st.image("./reports/figures/price_map.png", caption="Property prices in New York")

    st.markdown("""
                Aqui podemos encontrar algunas zonas con mas densidad que otra gracias a poner un alfa bajo en los putnos.
                
                Sin embargo, para poder ver mejor la distribucion de precios, hemos creado un plot en valores logaritmicos.
                """)

    st.image("./reports/figures/price_log_map.png", caption="Property prices in New York (log scale)")

with st.expander("Data Clearning", expanded=False):
    st.markdown("""
                El siguiente paso consiste en limpiar los datos para eliminar outliers y valores erroneos.
                
                Para esto, empezamos haciendo una correlacion entre las variables numericas del dataset con una matriz de calor.
                """)
    
    with st.expander("Correlation Matrix Before Cleaning", expanded=False):
        st.image("./reports/figures/correlation_matrix_before_cleaning.png", caption="Correlation Matrix")
        
        st.markdown("""
                    Con este analisis podemos ver que existe una correlacion muy baja entre el PRICE y las otras variables numericas como BEDS, BATH o PROPERTY SQFT.
                    
                    A esto podemos annyadir un plot pairplot para ver las relaciones entre las variables numericas.
                    """)
        
        st.image("./reports/figures/pairplot_before_cleaning.png", caption="Pairplot Before Cleaning")
        
        st.markdown("""
                    Donde podemos ver por un lado datos muy agrupados y por otro lado datos muy dispersos.
                    """)        
    
    st.markdown("""
                A continuacion, hemos aplicado tecnicas de deteccion de outlier mediante IQR y hemos eliminado valores que consideramos erroneos. Por ejemplo, eliminamos propiedades con 0 habitaciones o 0 baños.
                ya que no tienen sentido en el contexto del problema.
                
                Ademas, dado que la distribucion de precios es una distribucion con mucha cola, hemos aplicado una transformacion logaritimica a la variable objetivo PRICE para normalizar su distribucion.
                """)
    with st.expander("Price Distribution Before and After Cleaning", expanded=False):
        st.image("./reports/figures/price_distribution_comparison.png", caption="Clean Price Distribution Linear vs Log")
        
    st.markdown("""
                Esta misma transformacion la hemos aplicado para la variable PROPERTY SQFT, ya que tambien tiene una distribucion con mucha cola.
                """)
    
    st.markdown("""
                Finalmente, volvemos a hacer un pairplot y una matriz de correlacion para ver como han cambiado las relaciones entre las variables numericas.
                """)
    
    with st.expander("Correlation Matrix After Cleaning", expanded=False): 
        st.image("./reports/figures/correlation_matrix_after_cleaning.png", caption="Correlation Matrix After Cleaning")
        
        st.markdown("""
                    En este caso, podemos ver que la correlacion entre PRICE_LOG y las otras variables numericas ha aumentado, especialmente con BATH, donde existe una correlacion relativamente fuerte.
                    
                    Ademas, podemos ver que la correlacion entre las variables numericas ha cambiado, por ejemplo, la correlacion entre BEDS y BATH ha disminuido relativamente.
                    
                    A esto podemos annyadir un plot pairplot para ver las relaciones entre las variables numericas.
                    """)
        
        st.image("./reports/figures/pairplot_after_cleaning.png", caption="Pairplot After Cleaning")
        
        st.markdown("""
                    Donde podemos ver que los datos estan mucho mas agrupados y las relaciones entre las variables son mas claras.
                    
                    En resumen, hemos limpiado los datos eliminando outliers y valores erroneos, y hemos aplicado una transformacion logaritmica a las variables objetivo y a una variable numerica para normalizar su distribucion.
                    
                    Esto nos ha permitido mejorar la calidad de los datos y facilitar el analisis posterior.
                    """)
    
with st.expander("Feature Engineering", expanded=False):
    st.markdown("""
                El siguiente paso consiste en crear nuevas variables a partir de las existentes para mejorar el rendimiento del modelo.
                """)
    
    with st.expander("Structural Features", expanded=False):
        st.markdown("""
                    Las features estructurales nos permiten tener una mejor idea de la calidad de la propiedad en funcion de su tamanyo y su distribucion interna.
                    
                    Estas features son:
                    - **PRICE_PER_SQFT_LOG**: Precio por pie cuadrado en escala logaritmica. Esta variable nos permite tener una mejor idea del precio relativo de la propiedad en funcion de su tamaño.
                    - **BEDS_PER_SQFT**: Numero de habitaciones por pie cuadrado.
                    - **BATH_PER_SQFT**: Numero de baños por pie cuadrado.
                    """)
        
    with st.expander("Geographical Features", expanded=False):
        st.markdown("""
                    Las features geograficas nos permiten tener una mejor idea de la localizacion de la propiedad en funcion de su distancia a puntos de interes. O dividir la ciudad en zonas.
                    
                    Estas features son:
                    - **DISTANCE_TO_MANHATTAN**: Distancia a Manhattan en millas. Esta variable nos permite tener una mejor idea de la localizacion de la propiedad en funcion de su proximidad a un punto de interes.
                    - **LOCALITY**: Localidad de la propiedad. Esta variable nos permite dividir la ciudad en zonas y analizar las diferencias entre ellas.
                    - **SUBLOCALITY**: Sub-localidad de la propiedad. Esta variable nos permite dividir la ciudad en zonas mas pequeñas y analizar las diferencias entre ellas.
                    """)
        
        st.markdown("""
                    En este punto nos tenemos que parar a estudiar la relevancia real de estas variables, ya que puede ser que el dataset no este correctamente balanceado o que los datos no sean relevantes (como pasa en este caso)
                    
                    Es por esto, vamos a pararnos a analizar 3 casos de clustering de las propiedades diferentes, primer, utilizando la localidad, segundo, utilizando la sublocalidad y tercero, utilizando un clustering KMeans.
                    """)
        
        with st.expander("Locality Clustering", expanded=False):
            st.markdown("""
                A continuacion se muestra un mapa de la ciudad de Nueva York con las diferentes localidades coloreadas y con los bordes marcados.
            """)
            st.image("./reports/figures/localities_map.png", caption="Localities in New York")
            
            st.markdown("""
                        Donde podemos ver que no tiene mucho sentido, ya que hay localidades como 'New York' que ocupan toda la ciudad, algo que parece deberse a un mal etiquetado de los datos.
                        """)
            
            st.markdown("""
                        Ademas, si vemos la distribucion de precios por localidad, vemos que los datos estan toalmente desbalanceados.
                        """)
            
            st.image("./reports/figures/price_log_by_locality.png", caption="Locality PRICE Log Distribution")
            
        with st.expander("SubLocality Clustering", expanded=False):
            st.markdown("""
                Despues de hacer clustering por Locality, hemos estudiado el mapa de la ciudad utilizando SubLocality.
            """)
            st.image("./reports/figures/sublocalities_map.png", caption="Localities in New York")
            
            st.markdown("""
                        Donde podemos ver que los datos ya tienen un pooc mas de sentidos, aunque siguen aparecneido datos como 'New York' que ocupan toda la ciudad y que dubren muchos datos, como vemos en la distribucion de los mismos.
                        """)
            
            st.image("./reports/figures/price_log_by_sublocality.png", caption="Locality Price LOG Distribution")
            
            st.markdown("""
                Aunque en este caso, los datos ya estan mejor balanceada, ya no vemos la localidad de 'New York' con muchos datos y el resto de sublocalidades casi sin datos. Aunque todavia hay sublocalidades como por ejemplo
                'The Bronx', 'Staten Island' o 'New York County' que tienen muchos menos datos que el resto.
                """)
        
        with st.expander("KMeans Clustering", expanded=False):
            st.markdown("""
                Finalmente, hemos aplicado un clustering KMeans para ver si podemos encontrar clusters mas relevantes en funcion de la localizacion de las propiedades.
                
                Para esto, hemos utilizado las coordenadas geograficas (LATITUDE y LONGITUDE) y hemos estandarizado los datos para que el clustering sea mas efectivo.
                
                El resultado del clustering es el siguiente:
            """)
            st.image("./reports/figures/kmeans_clusters.png", caption="KMeans Clustering")
            
            st.markdown("""
                        Donde podemos ver que los clusters estan mucho mejor definidos y que no hay ningun cluster que ocupe toda la ciudad.
                        
                        Gracias a esto, podemos ver los datos con relacion al precio en funcion de los clusters creados.
                        """)
            
            st.image("./reports/figures/kmeans_clusters_polygons.png", caption="Price by KMeans Clusters")
            
            st.markdown("""
                        Donde podemos ver zonas con precios mas altos y zonas con precios mas bajos, y los datos estan mucho mejor balanceados (como se puede ver en la distribucion siguiente).
                        """)
            
            st.image("./reports/figures/price_log_by_cluster.png", caption="KMeans Price LOG Distribution")
            
        st.markdown("""
                    En resumen, hemos creado nuevas variables a partir de las existentes para mejorar el rendimiento del modelo.
                    
                    Sin embargo, no todas las variables creadas son relevantes para el problema, por lo que es importante analizar su relevancia antes de utilizarlas en el modelo.
                    
                    En este caso, hemos visto que las variables de Locality y SubLocality no son relevantes debido a un mal etiquetado de los datos, mientras que la variable de KMeans si es relevante debido a que los clusters estan mejor definidos y los datos estan mejor balanceados.
                    """)