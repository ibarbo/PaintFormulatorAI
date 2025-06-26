import streamlit as st
import pandas as pd
import joblib
import os
import shap
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import streamlit_shap as st_shap # Importamos la librería

# --- Configuración de Rutas ---
PREPROCESSOR_PATH = os.path.join('data', 'processed', 'preprocessor.joblib')
MODEL_PATH = os.path.join('models', 'random_forest_model.joblib')

# --- Cargar el Preprocesador y el Modelo ---
@st.cache_resource # Carga estos recursos una sola vez para mejorar el rendimiento
def load_resources():
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        # Crear el SHAP Explainer aquí, una vez que el modelo esté cargado
        explainer = shap.TreeExplainer(model)
        return preprocessor, model, explainer
    except FileNotFoundError as e:
        st.error(f"Error al cargar recursos. Asegúrate de que los archivos estén en las rutas correctas: {e}")
        st.stop() # Detiene la ejecución si no se encuentran los archivos

preprocessor, model, explainer = load_resources()

# --- Título de la Aplicación ---
st.set_page_config(layout="wide") # Opcional: para que la aplicación ocupe más ancho
st.title('🎨 Predictor de Éxito de Formulaciones de Pintura')
st.write('Sube un archivo CSV con tus nuevas formulaciones para obtener predicciones de éxito y análisis de interpretabilidad.')

# --- Carga de Archivo CSV ---
uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type=["csv"])

if uploaded_file is not None:
    try:
        # Leer el CSV
        df_input = pd.read_csv(uploaded_file)
        st.write("### Vista Previa de las Formulaciones Cargadas:")
        st.dataframe(df_input.head())

        # --- Preprocesamiento de los Datos ---
        st.write("### Preprocesando datos...")
        
        original_features = df_input.columns.tolist()

        dummy_transformed_features_array = preprocessor.transform(df_input.head(1))
        
        feature_names_transformed = []
        try:
            feature_names_transformed = preprocessor.get_feature_names_out().tolist()
        except AttributeError:
            st.warning("No se pudo obtener los nombres de las características transformadas automáticamente (get_feature_names_out()). Generando nombres genéricos. La interpretabilidad SHAP podría ser menos detallada para características categóricas transformadas (One-Hot Encoding).")
            feature_names_transformed = [f'feature_{i}' for i in range(dummy_transformed_features_array.shape[1])]
        
        if len(feature_names_transformed) != dummy_transformed_features_array.shape[1]:
            st.error(f"¡Error Crítico en nombres de características! El número de nombres ({len(feature_names_transformed)}) no coincide con el número de columnas transformadas ({dummy_transformed_features_array.shape[1]}). Esto causará problemas en SHAP. Por favor, revisa tu preprocesador. Forzando nombres genéricos para intentar continuar.")
            feature_names_transformed = [f'feature_{i}' for i in range(dummy_transformed_features_array.shape[1])]


        X_processed_array = preprocessor.transform(df_input)
        X_processed_df_for_shap = pd.DataFrame(X_processed_array, columns=feature_names_transformed)

        st.success("¡Datos preprocesados exitosamente!")

        # --- Realizar Predicciones ---
        st.write("### Realizando predicciones...")
        predictions = model.predict(X_processed_array)
        predictions_proba = model.predict_proba(X_processed_array)[:, 1]

        # --- Mostrar Resultados ---
        st.write("### Resultados de la Predicción:")

        df_results = df_input.copy()
        df_results['Predicción_Exito_Binario'] = predictions
        df_results['Probabilidad_Exito'] = predictions_proba.round(4)
        df_results['Predicción_Exito_Etiqueta'] = df_results['Predicción_Exito_Binario'].map({1: 'Éxito', 0: 'Falla'})

        df_display_results = df_results[['Predicción_Exito_Etiqueta', 'Probabilidad_Exito'] + original_features]
        st.dataframe(df_display_results)

        @st.cache_data 
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_download = convert_df_to_csv(df_results)
        st.download_button(
            label="Descargar resultados como CSV",
            data=csv_download,
            file_name="predicciones_formulaciones.csv",
            mime="text/csv",
        )

        # --- Sección para Interpretabilidad (SHAP) ---
        st.markdown("---")
        st.subheader("Análisis de Interpretabilidad (SHAP) para una Formulación Específica")
        st.write("Selecciona una formulación de la tabla para ver qué factores influyeron en su predicción.")

        df_input_with_index = df_input.reset_index(drop=True)
        selected_row_index = st.selectbox(
            "Selecciona el número de fila (desde 0) para analizar:",
            df_input_with_index.index
        )

        if selected_row_index is not None:
            st.write(f"Analizando la formulación en la fila: **{selected_row_index}**")
            
            selected_formula_for_shap = X_processed_df_for_shap.iloc[[selected_row_index]] 

            shap_values_raw = explainer.shap_values(selected_formula_for_shap)

            if isinstance(shap_values_raw, list) and len(shap_values_raw) > 1:
                shap_values_to_plot = shap_values_raw[1] 
            else:
                shap_values_to_plot = shap_values_raw 

            if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
                expected_value_to_plot = explainer.expected_value[1] 
            else:
                expected_value_to_plot = explainer.expected_value
            
            # --- AGREGADO DE DEBUGGING: WATERFALL PLOT (CORRECCIÓN DE DIMENSIONES) ---
            st.write("### SHAP Waterfall Plot (para depuración - si hay errores con Force Plot):")
            
            values_for_explanation = None
            if shap_values_to_plot.ndim == 1:
                values_for_explanation = shap_values_to_plot
            elif shap_values_to_plot.ndim == 2 and shap_values_to_plot.shape[0] == 1:
                values_for_explanation = shap_values_to_plot[0]
            elif shap_values_to_plot.ndim == 3 and shap_values_to_plot.shape[0] == 1 and shap_values_to_plot.shape[2] > 1:
                st.warning("DEBUG: shap_values_to_plot es 3D. Tomando [0, :, 1] para la clase 'Éxito'.")
                values_for_explanation = shap_values_to_plot[0, :, 1]
            else:
                st.error(f"DEBUG: Forma inesperada para shap_values_to_plot: {shap_values_to_plot.shape}. No se puede crear la Explicación para Waterfall Plot.")
                raise ValueError("Forma inesperada de valores SHAP para el waterfall plot.")

            shap_explanation = shap.Explanation(
                values=values_for_explanation,
                base_values=expected_value_to_plot,
                data=selected_formula_for_shap.iloc[0].values,
                feature_names=selected_formula_for_shap.columns.tolist()
            )
            
            plt.clf()
            plt.figure(figsize=(12, 7))
            
            shap.plots.waterfall(shap_explanation, show=False)
            
            try:
                if selected_row_index < len(predictions_proba):
                    plt.gca().set_title(f"Waterfall Plot para fila {selected_row_index} (Prob. Éxito: {predictions_proba[selected_row_index]:.2f})")
                else:
                    plt.gca().set_title(f"Waterfall Plot para fila {selected_row_index}")
            except Exception as e_title:
                st.warning(f"No se pudo establecer el título del Waterfall Plot directamente. Esto puede ser normal si SHAP ya lo generó. Error: {e_title}")

            st.pyplot(plt)
            st.write("---")
            # --- FIN DEBUGGING WATERFALL PLOT ---

            st.write("### SHAP Force Plot para la formulación seleccionada:")
            
            try:
                # Generamos el objeto de plot interactivo usando la función shap.force_plot original
                # Esta función devuelve un objeto que contiene el HTML y el JavaScript
                shap_plot_object = shap.force_plot(
                    expected_value_to_plot,
                    values_for_explanation, # Esto es crucial, debe ser 1D
                    selected_formula_for_shap.iloc[0] # Esto también debe ser 1D (una Serie de Pandas)
                )
                
                # Ahora usamos st_shap.st_shap() para renderizar este objeto en Streamlit
                # Esta es la función principal de streamlit-shap para mostrar los plots de SHAP.
                st_shap.st_shap(shap_plot_object)

            except Exception as force_plot_error:
                st.error(f"Error al generar o mostrar el SHAP Force Plot interactivo: {force_plot_error}")
                st.info("Asegúrate de haber instalado 'streamlit-shap' (`pip install streamlit-shap`) y de que tu versión de SHAP y Streamlit sean compatibles.")

            st.write("El gráfico muestra cómo cada característica (en rojo si empuja hacia el éxito, en azul si empuja hacia la falla) contribuye a la predicción final.")
            st.write("El valor base (Expected Value) es la predicción promedio sin considerar ninguna característica específica. El valor de salida (Output Value) es la probabilidad de éxito predicha para esta formulación.")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo o generar el análisis SHAP. Asegúrate de que el CSV tenga el formato correcto y las columnas esperadas. También revisa la compatibilidad del preprocesador y el modelo. Error: {e}")

else:
    st.info("Por favor, sube un archivo CSV para comenzar con las predicciones y el análisis de interpretabilidad.")