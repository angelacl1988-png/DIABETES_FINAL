import streamlit as st
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from graphviz import Digraph
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,  recall_score
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats
from prince import MCA


plt.style.use("dark_background")
sns.set_theme(style="darkgrid", palette="deep")

# === Título general ===
st.title("NHANES DIABETES Dashboard (2021-2023)")

# === Cargar base ===
df = pd.read_csv("nhanes_limpio.csv")


# === Sidebar Filtros ===
# === Filtros en sidebar ===
st.sidebar.header("🔎 Filtros")

# Edad
min_age, max_age = int(df["Edad en años"].min()), int(df["Edad en años"].max())
age_range = st.sidebar.slider("Edad (años):", min_age, max_age, (min_age, max_age))

# Sexo
gender_options = df["Sexo"].dropna().unique().tolist()
gender_filter = st.sidebar.multiselect("Sexo:", options=gender_options, default=gender_options)

# Raza / Etnia
race_options = df["Raza/etnia"].dropna().unique().tolist()
race_filter = st.sidebar.multiselect("Raza / Etnia:", options=race_options, default=race_options)

# IMC
min_bmi, max_bmi = float(df["Índice de masa corporal"].min()), float(df["Índice de masa corporal"].max())
bmi_range = st.sidebar.slider("Índice de masa corporal (BMI):", min_bmi, max_bmi, (min_bmi, max_bmi))

# Índice de pobreza (PIR)
poverty_range = st.sidebar.slider(
    "Índice de pobreza (PIR):",
    float(df["Índice de pobreza familiar (PIR)"].min()),
    float(df["Índice de pobreza familiar (PIR)"].max()),
    (
        float(df["Índice de pobreza familiar (PIR)"].min()),
        float(df["Índice de pobreza familiar (PIR)"].max())
    )
)

# === Aplicar filtros SOLO para Tab1 y Tab2 ===
filtered_df = df[
    (df["Edad en años"].between(age_range[0], age_range[1])) &
    (df["Sexo"].isin(gender_filter)) &
    (df["Raza/etnia"].isin(race_filter)) &
    (df["Índice de masa corporal"].between(bmi_range[0], bmi_range[1])) &
    (df["Índice de pobreza familiar (PIR)"].between(poverty_range[0], poverty_range[1]))
]


# === Pestañas ===
tab1, tab2, tab3, tab4= st.tabs(["Revisión inicial/criterios de selección","🔎 Indicadores iniciales",  "Reducción de dimensiones", "Técnicas de selección de variables"])


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# === Tab1 ===
with tab1:
    # === Texto introductorio en recuadro claro con letra oscura ===
    st.markdown("""
    <div style="background-color:#f9f9f9; padding:20px; border-radius:10px; color:#1a1a1a;">
        <strong>NHANES:</strong> El <strong>National Health and Nutrition Examination Survey (NHANES)</strong> es un programa de estudios de salud realizado por el <em>National Center for Health Statistics (NCHS)</em> de los <em>Centers for Disease Control and Prevention (CDC)</em> de Estados Unidos.<br><br>
        Su objetivo es evaluar el estado de salud y nutrición de la población estadounidense mediante un diseño muestral representativo a nivel nacional.<br><br>
        El estudio combina una <strong>entrevista en el hogar</strong> —en la que se recogen datos demográficos, socioeconómicos, dietarios y de salud— con un <strong>examen físico y pruebas de laboratorio</strong> realizados en un <em>Mobile Examination Center (MEC)</em>, que es una unidad clínica móvil equipada para realizar evaluaciones estandarizadas.<br><br>
        Los datos se recogen de manera continua y se publican en ciclos de <strong>dos años</strong>, lo que permite analizar tendencias en salud a lo largo del tiempo. NHANES incluye participantes de todas las edades y etnias, y sus resultados son ampliamente utilizados para la <strong>vigilancia epidemiológica</strong>, la <strong>investigación clínica</strong> y la <strong>formulación de políticas públicas en salud</strong>.

    </div>
    """, unsafe_allow_html=True)


  

   # === Diagrama de flujo ===
    st.subheader("Diagrama de Flujo del Proceso de Selección de Datos")
    dot = Digraph(comment='Flujo de Selección de Datos', format='png')

    dot.node('A', 'Base 2021-2023\nn = 11933', shape='box', style='filled', color='lightblue')
    dot.node('B', 'Incluidos\nn = 6,296', shape='box', style='filled', color='lightgreen')
    dot.node('C', 'Base final\n6072 registros', shape='box', style='filled', color='lightgreen')
    dot.node('D', 'Se excluyeron:\nNo contestaron encuesta MET (n=3073)\nMenores de 18 años (n=2523)\nEmbarazadas (n=41)', shape='box', style='filled', color='orange')
    dot.node('E', 'Se excluyeron sujetos con valores perdidos en la variable objetivo (n=224)', shape='box', style='filled', color='red')

    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('B', 'D', constraint='false')
    dot.edge('C', 'E', constraint='false')

    st.graphviz_chart(dot)

    # === Información de la base ===
    st.info(f"📌 La base de datos tiene **{df.shape[1]} variables** y **{df.shape[0]} registros**.")

    # Muestra un sample
    st.dataframe(df.head(5))

    # --- Valores faltantes ---
    faltantes = df.isnull().mean() * 100
    faltantes_df = faltantes.reset_index()
    faltantes_df.columns = ["Variable", "% Valores faltantes"]
    faltantes_df["% Valores faltantes"] = faltantes_df["% Valores faltantes"].round(2)
    faltantes_df["Nulos"] = df.isnull().sum().values
    faltantes_df = faltantes_df.sort_values(by="% Valores faltantes", ascending=False).reset_index(drop=True)

    # --- Gráfica interactiva de Plotly ---
    st.subheader("📊 Visualización interactiva de valores faltantes")
    fig_na = px.bar(
        faltantes_df.head(15),
        x="% Valores faltantes",
        y="Variable",
        orientation='h',
        text="% Valores faltantes",
        color="% Valores faltantes",
        color_continuous_scale="Viridis",
        hover_data={"Variable": True, "% Valores faltantes": True, "Nulos": True}
    )
    fig_na.update_layout(
        title="Top 15 variables con más valores faltantes",
        xaxis_title="% de valores faltantes",
        yaxis_title="Variable",
        yaxis={'categoryorder':'total ascending'}
    )
    st.plotly_chart(fig_na, use_container_width=True)

    st.markdown(
        """
        ### Observaciones sobre los datos  
        
        Se encontró que las variables con mayor cantidad de valores perdidos (>80%) fueron:  
        - **Duración de estancia en EEUU**  
        - **Uso actual de insulina**  
        
        Sin embargo, esta última variable depende de si el paciente es diabético, lo cual justifica la cantidad de valores perdidos.  
    
        Más adelante, estas variables serán eliminadas como parte del proceso de **reducción de dimensiones** y **selección de variables**.
        """
    )
    


## ------------------------------------------------
# TAB 2: Explorador
# ------------------------------------------------
with tab2:

          # ============================
    # Cálculo de prevalencias
    # ============================
    import streamlit as st
    import pandas as pd
    import numpy as np
    
    # Total de pacientes
    total_pacientes = len(df)
    
    # Evitar dividir por 0
    if total_pacientes > 0:
        # Diabetes
        casos_diabetes = (filtered_df["Diagnóstico médico de diabetes"] == "Sí").sum()
        prevalencia_diabetes = (casos_diabetes / total_pacientes) * 100
    
        # Prediabetes
        casos_prediabetes = (filtered_df["Diagnóstico médico de prediabetes"] == "Sí").sum()
        prevalencia_prediabetes = (casos_prediabetes / total_pacientes) * 100
    
        # Uso de insulina
        casos_insulina = filtered_df["Uso actual de insulina"] == "Sí").sum()
        prevalencia_insulina = (casos_insulina / total_pacientes) * 100
    else:
        casos_diabetes = casos_prediabetes = casos_insulina = 0
        prevalencia_diabetes = prevalencia_prediabetes = prevalencia_insulina = 0
    

    
    # ============================
    # Métricas en columnas
    # ============================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Prediabetes",
            value=f"{prevalencia_prediabetes:.2f}%",
            delta=f"{casos_prediabetes}/{total_pacientes} casos"
        )
    
    with col2:
        st.metric(
            label="Diabetes",
            value=f"{prevalencia_diabetes:.2f}%",
            delta=f"{casos_diabetes}/{total_pacientes} casos"
        )
    
    with col3:
        st.metric(
            label="Uso de Insulina",
            value=f"{prevalencia_insulina:.2f}%",
            delta=f"{casos_insulina}/{total_pacientes} casos"
        )



    st.subheader("📊 Análisis interactivo de variables vs Diabetes")

    # === Selección de variable ===
    variable_seleccionada = st.selectbox(
        "Selecciona una variable para visualizar su relación con Diabetes",
        options=[col for col in df.columns if col not in ["Diagnóstico médico de diabetes", "SEQN"]]
    )

    # === Detectar tipo de variable y crear gráfico ===
    
    # --- Filtrar exclusiones ---
    excluir = ["Diagnóstico médico de prediabetes", "Uso actual de insulina"]

    if variable_seleccionada not in excluir:

        if df[variable_seleccionada].dtype in ['int64', 'float64']:
            # Boxplot interactivo con Plotly
            fig = px.box(
                df,
                x="Diagnóstico médico de diabetes",
                y=variable_seleccionada,
                color="Diagnóstico médico de diabetes",
                points="all",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title=f"Distribución de {variable_seleccionada} según Diabetes"
            )
            
            # --- Prueba estadística ---
            grupos = df.groupby("Diagnóstico médico de diabetes")[variable_seleccionada].apply(list)
            if len(grupos) == 2:
                stat, p = stats.ttest_ind(grupos["Sí"], grupos["No"], nan_policy='omit')
                st.write(f"**Prueba t de Student:** p-value = {p:.4f}")
            else:
                stat, p = stats.f_oneway(*[v for v in grupos])
                st.write(f"**ANOVA:** p-value = {p:.4f}")

        else:
            # Diagrama de barras
            conteo = df.groupby([variable_seleccionada, "Diagnóstico médico de diabetes"]).size().reset_index(name="Frecuencia")
            fig = px.bar(
                conteo,
                x=variable_seleccionada,
                y="Frecuencia",
                color="Diagnóstico médico de diabetes",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title=f"Distribución de {variable_seleccionada} vs Diabetes"
            )
            
            # --- Prueba estadística ---
            tabla = pd.crosstab(df[variable_seleccionada], df["Diagnóstico médico de diabetes"])
            chi2, p, dof, expected = stats.chi2_contingency(tabla)
            st.write(f"**Chi-cuadrado:** p-value = {p:.4f}")

        # Mostrar gráfico interactivo
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(f"La variable **{variable_seleccionada}** está excluida del análisis.")
        


# TAB 3 - PCA / MCA
# =====================
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# ======================================================
# TAB 3: PCA / MCA
# ======================================================
with tab3:
    tab_pca, tab_mca = st.tabs(["PCA / Numéricas", "MCA / Categóricas"])

    # ======================================================
    # SUBTAB PCA
    # ======================================================
    with tab_pca:
    

        # Umbral dinámico
        VAR_THRESHOLD = st.slider("Umbral de selección de varianza acumulada:", 
                                  0.5, 0.99, 0.80, 0.01)

        # Selección y limpieza de variables numéricas
        vars_excluir = ["SEQN", "Diagnóstico médico de diabetes", 
                        "Diagnóstico médico de prediabetes", "Uso actual de insulina"]
        X = df.drop(columns=vars_excluir, errors="ignore")
        X_num = X.select_dtypes(include=[np.number])

        # Excluir variables con >80% de NaN
        missing_frac = X_num.isna().mean()
        X_num = X_num.loc[:, missing_frac <= 0.8]

           # === Información de la base ===
        st.info(f"📌 La base de datos tiene **{X_num.shape[1]} variables** y **{X_num.shape[0]} registros**.")

        # Preprocesamiento
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        Xn_tr = num_pipe.fit_transform(X_num)

        # PCA completo
        pca_full = PCA().fit(Xn_tr)
        var_ratio = pca_full.explained_variance_ratio_
        var_cum = np.cumsum(var_ratio)

        # Nº mínimo de componentes para alcanzar el umbral
        k_pca = int(np.searchsorted(var_cum, VAR_THRESHOLD) + 1)

        # PCA final
        pca = PCA(n_components=k_pca).fit(Xn_tr)
        Z_num = pca.transform(Xn_tr)

        PC_cols = [f"PC{i}" for i in range(1, k_pca+1)]
        DF_PCA = pd.DataFrame(Z_num, columns=PC_cols, index=X_num.index)

        st.write(f"[PCA] Componentes seleccionados: **{k_pca}** (umbral={VAR_THRESHOLD:.0%})")

        # Tabla de varianzas
        df_var = pd.DataFrame({
            "Componente": [f"PC{i}" for i in range(1, len(var_cum)+1)],
            "Varianza individual": np.round(var_ratio, 4),
            "Varianza acumulada": np.round(var_cum, 4)
        })
        st.dataframe(df_var)

        # =====================
        # Scree plot interactivo con Plotly
        # =====================
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Scatter(
            x=list(range(1, len(var_cum)+1)),
            y=var_cum,
            mode='lines+markers',
            name='Varianza acumulada'
        ))
        fig_scree.add_hline(y=VAR_THRESHOLD, line_dash="dash", line_color="red",
                            annotation_text=f"Umbral {VAR_THRESHOLD*100:.0f}%", annotation_position="top right")
        fig_scree.add_vline(x=k_pca, line_dash="dash", line_color="green",
                            annotation_text=f"{k_pca} comp.", annotation_position="top left")
        fig_scree.update_layout(
            title="Scree plot - PCA",
            xaxis_title="Número de componentes",
            yaxis_title="Varianza explicada acumulada",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white")
        )
        st.plotly_chart(fig_scree, use_container_width=True)

        # =====================
        # INTERACTIVO: contribución de variables con Plotly
        # =====================
        st.write("### 🔎 Explora las variables que más aportan a cada componente")

        loadings = pca.components_.T
        df_loadings = pd.DataFrame(
            loadings,
            index=X_num.columns,
            columns=[f"PC{i}" for i in range(1, k_pca+1)]
        )

        pc_choice = st.selectbox("Selecciona un componente principal:", df_loadings.columns)
        top_n = st.slider("Número de variables a mostrar:", 5, 20, 10)

        st.markdown(f"#### {pc_choice} (varianza explicada: {pca.explained_variance_ratio_[int(pc_choice[2:])-1]*100:.2f}%)")
        top_vars = df_loadings[pc_choice].abs().sort_values(ascending=False).head(top_n)

        # Gráfico de barras
        fig_top_vars = px.bar(
            top_vars.sort_values(),
            x=top_vars.sort_values().values,
            y=top_vars.sort_values().index,
            orientation='h',
            color=top_vars.sort_values().values,
            color_continuous_scale="Viridis",
            labels={"y": "Variable", "x": "Contribución (|loading|)"}
        )
        fig_top_vars.update_layout(
            title=f"Top {top_n} variables que aportan a {pc_choice}",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_top_vars, use_container_width=True)

        # =====================
        # BIPLOT INTERACTIVO PCA
        # =====================
        if k_pca >= 2:
            st.subheader("📌 Biplot interactivo (segmentado por Diabetes)")

            comp_options = [f"PC{i}" for i in range(1, k_pca+1)]
            col1, col2 = st.columns(2)
            pc_x = col1.selectbox("Eje X:", comp_options, index=0)
            pc_y = col2.selectbox("Eje Y:", comp_options, index=1)

            ix_x = int(pc_x[2:]) - 1
            ix_y = int(pc_y[2:]) - 1

            pcs_df = pd.DataFrame({
                pc_x: Z_num[:, ix_x],
                pc_y: Z_num[:, ix_y],
                "Diabetes": df["Diagnóstico médico de diabetes"]
            })

            fig = px.scatter(
                pcs_df, x=pc_x, y=pc_y, color="Diabetes",
                labels={pc_x: f"{pc_x} ({pca.explained_variance_ratio_[ix_x]*100:.2f}%)",
                        pc_y: f"{pc_y} ({pca.explained_variance_ratio_[ix_y]*100:.2f}%)"},
                opacity=0.7
            )
            fig.update_layout(
                title=f"Biplot PCA ({pc_x} vs {pc_y}) - Segmentado por Diabetes",
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white"),
                legend=dict(itemsizing="constant", orientation="h", y=-0.2)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ El Biplot interactivo requiere al menos 2 componentes principales.")

        # =====================
        # Dataset final con PCs + objetivo
        # =====================
        TARGET_COL = "Diagnóstico médico de diabetes"  # Columna objetivo
        DF_PCA_final = pd.concat(
            [df[[TARGET_COL]].reset_index(drop=True),
             DF_PCA.reset_index(drop=True)],
            axis=1
        )

        # Guardar en CSV
        DF_PCA_final.to_csv("pca_componentes.csv", index=False)
        st.success(f"✅ Guardado **pca_componentes.csv** con shape: {DF_PCA_final.shape}")

        # Mostrar vista previa
        st.dataframe(DF_PCA_final.head())

       # ======================================================
       # ======================================================
    # SUBTAB MCA
    # ======================================================

    with tab_mca:
           
        # Selección y limpieza de categóricas
       
        X_cat = X.select_dtypes(exclude=[np.number])

           # === Información de la base ===
        st.info(f"📌 La base de datos tiene **{X_cat.shape[1]} variables** y **{X_cat.shape[0]} registros**.")

        # Matriz disyuntiva (todas las categorías)
        X_disc = pd.get_dummies(X_cat, drop_first=False)
        # Mostrar tamaño (n_filas, n_columnas)
        st.write(f"📐 Tamaño de la matriz disyuntiva: {X_disc.shape[0]} filas x {X_disc.shape[1]} columnas")

        # Umbral dinámico
        VAR_THRESHOLD = st.slider("Umbral de selección de inercia acumulada:", 
                                  0.5, 0.99, 0.80, 0.01)


        # Ajustar MCA
        import mca
        m = mca.MCA(X_disc, benzecri=True)

        eig = np.array(m.L, dtype=float).ravel()
        inertia = eig / eig.sum()
        inertia_cum = np.cumsum(inertia)

        k_mca = int(np.searchsorted(inertia_cum, VAR_THRESHOLD) + 1)

        st.write(f"[MCA] Dimensiones seleccionadas: **{k_mca}** (umbral={VAR_THRESHOLD:.0%})")

        # Tabla de inercias
        df_inertia = pd.DataFrame({
            "Dimensión": [f"DIM{i}" for i in range(1, len(inertia_cum)+1)],
            "Inercia individual": np.round(inertia, 4),
            "Inercia acumulada": np.round(inertia_cum, 4)
        })
        st.dataframe(df_inertia)

        # =====================
        # Scree plot interactivo con Plotly
        # =====================
        fig_scree_mca = go.Figure()
        fig_scree_mca.add_trace(go.Scatter(
            x=list(range(1, len(inertia_cum)+1)),
            y=inertia_cum,
            mode='lines+markers',
            name='Inercia acumulada'
        ))
        fig_scree_mca.add_hline(y=VAR_THRESHOLD, line_dash="dash", line_color="red",
                                annotation_text=f"Umbral {VAR_THRESHOLD*100:.0f}%", annotation_position="top right")
        fig_scree_mca.add_vline(x=k_mca, line_dash="dash", line_color="green",
                                annotation_text=f"{k_mca} dim.", annotation_position="top left")
        fig_scree_mca.update_layout(
            title="Scree plot - MCA",
            xaxis_title="Número de dimensiones",
            yaxis_title="Inercia acumulada",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white")
        )
        st.plotly_chart(fig_scree_mca, use_container_width=True)

        # =====================
        # Coordenadas de individuos
        # =====================
        Fs = m.fs_r(N=k_mca)
        DF_MCA = pd.DataFrame(Fs, index=X_cat.index,
                              columns=[f"DIM{i}" for i in range(1, k_mca+1)])

        # =====================
        # Contribución de variables (categorías) a las dimensiones
        # =====================
        mass = X_disc.mean(axis=0).values
        G = m.fs_c(N=k_mca)
        eig_k = eig[:k_mca]   # usar solo los primeros k_mca autovalores
        ctr = (mass[:, None] * (G**2)) / eig_k[None, :]

        ctr_pct = ctr / ctr.sum(axis=0, keepdims=True)

        DIM_cols = [f"DIM{i}" for i in range(1, k_mca+1)]
        DF_ctr_cat = pd.DataFrame(ctr_pct, index=X_disc.columns, columns=DIM_cols)

        st.write("### 📌 Contribución de categorías a las dimensiones")
        st.dataframe(DF_ctr_cat.head(20))  # mostrar primeras 20 filas

        # =====================
        # Dataset final con DIMs + objetivo
        # =====================
        TARGET_COL = "Diagnóstico médico de diabetes"
        DF_MCA_final = pd.concat(
            [df[[TARGET_COL]].reset_index(drop=True),
             DF_MCA.reset_index(drop=True)],
            axis=1
        )

        DF_MCA_final.to_csv("mca_dimensiones.csv", index=False)
        st.success(f"✅ Guardado **mca_dimensiones.csv** con shape: {DF_MCA_final.shape}")
        st.dataframe(DF_MCA_final.head())

        # =====================
        # Dataset final conjunto PCA + MCA + objetivo
        # =====================
        if "DF_PCA_final" in locals():
            DF_final = pd.concat([DF_PCA_final.reset_index(drop=True),
                                  DF_MCA.reset_index(drop=True)], axis=1)

            DF_final.to_csv("pca_mca_concat.csv", index=False)
            st.success(f"✅ Guardado **pca_mca_concat.csv** con shape: {DF_final.shape}")
            st.dataframe(DF_final.head())
        else:
            st.warning("⚠️ Aún no has corrido el bloque PCA para generar DF_PCA_final.")


# ------------------------------------------------
# TAB 4: Selección de Variables
# ------------------------------------------------
with tab4:

    ### Selección de variables y cobertura del 90% de la importancia

    import streamlit as st

    st.markdown("""
    ### Selección de variables y cobertura del 90% de la importancia
    
    En el análisis de selección de características se aplicaron tres enfoques diferentes para identificar las variables más relevantes en la predicción:
    
    1. **Filtrado (Chi²)**  
       - Evalúa la **asociación estadística** entre cada variable independiente y la variable objetivo de manera individual.  
       - Se seleccionan las variables que muestran mayor dependencia con la salida.  
       - En este caso, **solo 4 variables** fueron necesarias para cubrir el **90% de la importancia acumulada**.  
       - Indica que unas pocas variables tienen un peso muy fuerte en la relación con el desenlace.
    
    2. **Incrustado (Random Forest)**  
       - Utiliza modelos de aprendizaje automático (árboles de decisión en este caso) que asignan una **importancia** a cada variable durante el entrenamiento.  
       - Este método considera **interacciones** entre variables y relaciones no lineales.  
       - Se necesitaron **25 variables** para cubrir el **90% de la importancia**, lo cual refleja que el modelo distribuye la relevancia entre un mayor número de predictores.
    
    3. **Envoltura (RFECV – Recursive Feature Elimination con validación cruzada)**  
       - Evalúa conjuntos de variables de forma iterativa, eliminando las menos relevantes en cada paso.  
       - Utiliza validación cruzada para garantizar que las variables seleccionadas realmente contribuyen a mejorar el rendimiento del modelo.  
       - Aquí fueron necesarias **18 variables** para alcanzar el **90% de la importancia acumulada**.  
     
    """)


    # --- Preprocesamiento ---
    TARGET_COL = "Diagnóstico médico de diabetes"
    vars_excluir = ["SEQN", "Diagnóstico médico de prediabetes", "Uso actual de insulina"]

    # Solo predictores
    X = df.drop(columns=vars_excluir + [TARGET_COL], errors="ignore")
    y = df[TARGET_COL].map({"Sí": 1, "No": 0})  # convertir a binario

    # --- Asegurar numéricas para filtrado y regresión ---
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num = X_num.fillna(X_num.median())  # imputación rápida

    # ============================
    # 1. Filtrado (SelectKBest - Chi2)
    # ============================
    from sklearn.feature_selection import SelectKBest, chi2

    selector = SelectKBest(score_func=chi2, k="all")
    selector.fit(X_num.abs(), y)  # abs porque chi2 no acepta negativos

    scores_filter = selector.scores_
    features = X_num.columns

    indices_filter = np.argsort(scores_filter)[::-1]
    sorted_scores_filter = scores_filter[indices_filter]
    sorted_features_filter = features[indices_filter]

    cumulative_filter = np.cumsum(sorted_scores_filter) / np.sum(sorted_scores_filter)
    cutoff_filter = np.searchsorted(cumulative_filter, 0.90) + 1
    selected_filter = sorted_features_filter[:cutoff_filter]

    # ============================
    # 2. Incrustado (Random Forest)
    # ============================
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_num, y)

    importances_embedded = rf.feature_importances_
    indices_embedded = np.argsort(importances_embedded)[::-1]
    sorted_importances_embedded = importances_embedded[indices_embedded]
    sorted_features_embedded = features[indices_embedded]

    cumulative_embedded = np.cumsum(sorted_importances_embedded) / np.sum(sorted_importances_embedded)
    cutoff_embedded = np.searchsorted(cumulative_embedded, 0.90) + 1
    selected_embedded = sorted_features_embedded[:cutoff_embedded]

    # ============================
    # 3. Envoltura (RFECV con LogisticRegression)
    # ============================
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import RFECV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)

    rfecv = RFECV(estimator=model, step=1, cv=5, scoring="accuracy")
    pipeline = Pipeline([("scaler", scaler), ("feature_selection", rfecv)])
    pipeline.fit(X_num, y)

    selected_wrap = X_num.columns[rfecv.support_]
    coefs = rfecv.estimator_.coef_.flatten()

    indices_wrap = np.argsort(np.abs(coefs))[::-1]
    abs_coefs_sorted = np.abs(coefs)[indices_wrap]
    cumulative_wrap = np.cumsum(abs_coefs_sorted) / np.sum(abs_coefs_sorted)
    cutoff_wrap = np.searchsorted(cumulative_wrap, 0.90) + 1
    selected_wrap_90 = selected_wrap[indices_wrap][:cutoff_wrap]

    # ============================
    # Mostrar resultados en Streamlit
    # ============================
    st.subheader("📌 Número de variables necesarias para cubrir el 90% de la importancia")
    st.write(f"- Filtrado (Chi2): **{cutoff_filter}** variables")
    st.write(f"- Incrustado (Random Forest): **{cutoff_embedded}** variables")
    st.write(f"- Envoltura (RFECV): **{cutoff_wrap}** variables")

    st.subheader("🔹 Variables seleccionadas por cada método")
    st.write("**Filtrado (Chi2):**", selected_filter.tolist())
    st.write("**Incrustado (Random Forest):**", selected_embedded.tolist())
    st.write("**Envoltura (RFECV):**", selected_wrap_90.tolist())

    # ============================
    # Gráficas comparativas con Plotly
    # ============================
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig_vars = make_subplots(rows=1, cols=3, subplot_titles=("Filtrado (Chi2)", "Incrustado (RF)", "Envoltura (RFECV)"))

    # --- Filtrado
    fig_vars.add_trace(
        go.Bar(x=sorted_features_filter, y=sorted_scores_filter, marker_color="skyblue"),
        row=1, col=1
    )
    fig_vars.add_shape(
        type="line", x0=cutoff_filter - 1, x1=cutoff_filter - 1,
        y0=0, y1=max(sorted_scores_filter),
        line=dict(color="red", dash="dash"),
        row=1, col=1
    )

    # --- Incrustado
    fig_vars.add_trace(
        go.Bar(x=sorted_features_embedded, y=sorted_importances_embedded, marker_color="lightgreen"),
        row=1, col=2
    )
    fig_vars.add_shape(
        type="line", x0=cutoff_embedded - 1, x1=cutoff_embedded - 1,
        y0=0, y1=max(sorted_importances_embedded),
        line=dict(color="red", dash="dash"),
        row=1, col=2
    )

    # --- Envoltura
    fig_vars.add_trace(
        go.Bar(x=selected_wrap[indices_wrap], y=abs_coefs_sorted, marker_color="salmon"),
        row=1, col=3
    )
    fig_vars.add_shape(
        type="line", x0=cutoff_wrap - 1, x1=cutoff_wrap - 1,
        y0=0, y1=max(abs_coefs_sorted),
        line=dict(color="red", dash="dash"),
        row=1, col=3
    )

    fig_vars.update_layout(
        title_text="🔎 Comparación de Importancia de Variables por Método",
        showlegend=False,
        height=500, width=1200
    )
    st.plotly_chart(fig_vars)

    # ============================
    # Evaluación de modelos según selección de variables
    # ============================
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    
    # Split común
    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=0.3, random_state=42, stratify=y
    )
    
    def evaluar_modelo(features, nombre):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train[features], y_train)
        y_pred = model.predict(X_test[features])
        y_prob = model.predict_proba(X_test[features])[:, 1]
    
        return {
            "Método": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "fpr_tpr": roc_curve(y_test, y_prob)
        }
    
    # Evaluar cada método
    resultados = []
    resultados.append(evaluar_modelo(selected_filter, "Filtrado (Chi2)"))
    resultados.append(evaluar_modelo(selected_embedded, "Incrustado (RF)"))
    resultados.append(evaluar_modelo(selected_wrap_90, "Envoltura (RFECV)"))
    
    # ============================
    # Gráfico de métricas comparativas
    # ============================
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    fig_metrics = go.Figure()
    
    for r in resultados:
        fig_metrics.add_trace(go.Bar(
            x=metrics,
            y=[r[m] for m in metrics],
            name=r["Método"]
        ))
    
    fig_metrics.update_layout(
        title="📊 Comparación de métricas por método de selección",
        barmode="group",
        yaxis=dict(title="Valor")
    )
    st.plotly_chart(fig_metrics)
    
    # ============================
    # Curvas ROC comparativas
    # ============================
    fig_roc = go.Figure()
    
    for r in resultados:
        fpr, tpr, _ = r["fpr_tpr"]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{r['Método']} (AUC={r['AUC']:.2f})"
        ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash", color="gray"),
        showlegend=False
    ))
    
    fig_roc.update_layout(
        title="📈 Curvas ROC comparativas",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        width=700, height=500
    )
    st.plotly_chart(fig_roc)


    st.markdown("""
    
        **Interpretación global:**  
    - El método **Chi² (filtrado)** es más restrictivo, seleccionando muy pocas variables clave.  
    - El método **Random Forest (incrustado)** considera interacciones complejas y requiere más variables.  
    - El **RFECV (envoltura)** busca un equilibrio entre simplicidad y rendimiento del modelo.  
    
    La elección final del conjunto de variables dependerá del **objetivo del estudio**:  
    - Si se busca **simplicidad y explicabilidad**, conviene usar un método filtrado.  
    - Si se prioriza **precisión predictiva**, los métodos incrustados o de envoltura suelen ser más adecuados.  
    """)
    



