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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Revisión inicial/criterios de selección","🔎 Indicadores iniciales",  "Reducción de dimensiones", "Selección de variables", "Comparación PCA_MCA vs RF"])


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


## ------------------------------------------------
# TAB 2: Explorador
# ------------------------------------------------
with tab2:

    # Función para calcular prevalencia y frecuencia
    def calcular_prevalencia_y_frecuencia(df, columna):
        df_filtrado = df[df[columna].notnull()]  # ignorar nulos
        total = len(df_filtrado)
        positivos = (df_filtrado[columna] == "Sí").sum()
        prevalencia = (positivos / total) * 100 if total > 0 else 0
        return prevalencia, positivos, total
    
    # Calcular prevalencias y frecuencias
    prevalencia_diabetes, casos_diabetes, total_diabetes = calcular_prevalencia_y_frecuencia(df, "Diagnóstico médico de diabetes")
    prevalencia_prediabetes, casos_prediabetes, total_prediabetes = calcular_prevalencia_y_frecuencia(df, "Diagnóstico médico de prediabetes")
    prevalencia_insulina, casos_insulina, total_insulina = calcular_prevalencia_y_frecuencia(df, "Uso actual de insulina")
    
    # Mostrar en Streamlit
    st.title("Prevalencias de Condiciones Médicas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Diabetes",
            value=f"{prevalencia_diabetes:.2f}%",
            delta=f"{casos_diabetes}/{total_diabetes} casos"
        )
    with col2:
        st.metric(
            label="Prediabetes",
            value=f"{prevalencia_prediabetes:.2f}%",
            delta=f"{casos_prediabetes}/{total_prediabetes} casos"
        )
    with col3:
        st.metric(
            label="Uso de Insulina",
            value=f"{prevalencia_insulina:.2f}%",
            delta=f"{casos_insulina}/{total_insulina} casos"
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

# TAB 4 - Selección de Variables
# ==============================
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

with tab4:
    st.subheader("📌 Selección de Variables: LASSO vs Random Forest")

    TARGET_COL = "Diagnóstico médico de diabetes"

    # --- Preparar dataset ---
    X = df.drop(columns=["SEQN", "Diagnóstico médico de prediabetes", "Uso actual de insulina"], errors="ignore")
    y = df[TARGET_COL].map({"Sí": 1, "No": 0})

    # Codificar categóricas
    X_encoded = pd.get_dummies(X.drop(columns=[TARGET_COL], errors="ignore"), drop_first=True)

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- Imputación y escalado ---
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # === LASSO ===
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    coef = pd.Series(lasso.coef_, index=X_encoded.columns)
    lasso_vars = coef[coef != 0].sort_values(key=abs, ascending=False)

    st.markdown("### 🔎 Variables seleccionadas por **LASSO**")
    st.dataframe(lasso_vars.head(20))

    fig_lasso = px.bar(
        lasso_vars.head(15).sort_values(),
        x=lasso_vars.head(15).sort_values().values,
        y=lasso_vars.head(15).sort_values().index,
        orientation="h",
        title="Top 15 coeficientes distintos de cero (LASSO)",
        color=lasso_vars.head(15).abs().values,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_lasso, use_container_width=True)

    # === Random Forest ===
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X_train_imp, y_train)

    importancias = pd.Series(rf.feature_importances_, index=X_encoded.columns)
    rf_vars = importancias.sort_values(ascending=False)

    st.markdown("### 🔎 Importancia de variables según **Random Forest**")
    st.dataframe(rf_vars.head(20))

    fig_rf = px.bar(
        rf_vars.head(15).sort_values(),
        x=rf_vars.head(15).sort_values().values,
        y=rf_vars.head(15).sort_values().index,
        orientation="h",
        title="Top 15 variables más importantes (Random Forest)",
        color=rf_vars.head(15).values,
        color_continuous_scale="Plasma"
    )
    st.plotly_chart(fig_rf, use_container_width=True)

    # === Comparación directa ===
    st.subheader("📊 Comparación LASSO vs RF")
    comparacion = pd.DataFrame({
        "LASSO (|coef|)": lasso_vars.reindex(X_encoded.columns).abs(),
        "Random Forest (importancia)": rf_vars.reindex(X_encoded.columns)
    }).fillna(0)

    fig_comp = px.scatter(
        comparacion,
        x="LASSO (|coef|)",
        y="Random Forest (importancia)",
        text=comparacion.index,
        size="Random Forest (importancia)",
        opacity=0.7,
        title="Comparación de importancia de variables: LASSO vs Random Forest"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # === Evaluación con curva ROC y métricas ===
    st.subheader("📈 Comparación de desempeño")

    # --- LASSO como regresión logística ---
    log_reg = LogisticRegression(max_iter=10000)
    X_train_sel = X_train_scaled[:, coef != 0]
    X_test_sel = X_test_scaled[:, coef != 0]

    log_reg.fit(X_train_sel, y_train)
    y_pred_lasso = log_reg.predict(X_test_sel)
    y_prob_lasso = log_reg.predict_proba(X_test_sel)[:, 1]

    # --- Random Forest ---
    y_pred_rf = rf.predict(X_test_imp)
    y_prob_rf = rf.predict_proba(X_test_imp)[:, 1]

    # --- Curva ROC ---
    fpr_lasso, tpr_lasso, _ = roc_curve(y_test, y_prob_lasso)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_lasso, y=tpr_lasso, mode="lines", name=f"LASSO (AUC={auc(fpr_lasso,tpr_lasso):.3f})"))
    fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines", name=f"Random Forest (AUC={auc(fpr_rf,tpr_rf):.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar", line=dict(dash="dash")))
    fig_roc.update_layout(title="Curva ROC - LASSO vs Random Forest", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    # --- Métricas ---
    metrics_df = pd.DataFrame({
        "Accuracy": [
            accuracy_score(y_test, y_pred_lasso),
            accuracy_score(y_test, y_pred_rf)
        ],
        "Precision": [
            precision_score(y_test, y_pred_lasso),
            precision_score(y_test, y_pred_rf)
        ],
        "Recall": [
            recall_score(y_test, y_pred_lasso),
            recall_score(y_test, y_pred_rf)
        ],
        "F1-Score": [
            f1_score(y_test, y_pred_lasso),
            f1_score(y_test, y_pred_rf)
        ],
        "ROC AUC": [
            roc_auc_score(y_test, y_prob_lasso),
            roc_auc_score(y_test, y_prob_rf)
        ]
    }, index=["LASSO", "Random Forest"])

    st.dataframe(metrics_df)  # <- ✅ ahora funciona sin Styler


# ======================================================
# TAB 5: Comparación PCA vs RandomForest
# ======================================================
with tab5:
    st.subheader("📊 Comparación PCA vs RandomForest")

    if 'DF_PCA_final' not in locals():
        st.warning("⚠️ Primero ejecuta el bloque PCA (TAB 3) para generar DF_PCA_final.")
    else:
        # Preparar datos PCA
        X_pca = DF_PCA_final.drop(columns=[TARGET_COL])
        y_pca = LabelEncoder().fit_transform(DF_PCA_final[TARGET_COL])
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
            X_pca, y_pca, test_size=0.3, stratify=y_pca, random_state=42
        )

        rf_pca = RandomForestClassifier(n_estimators=500, random_state=42)
        rf_pca.fit(X_train_pca, y_train_pca)
        y_pred_pca = rf_pca.predict_proba(X_test_pca)[:, 1]

        fpr_pca, tpr_pca, _ = roc_curve(y_test_pca, y_pred_pca)
        auc_pca = auc(fpr_pca, tpr_pca)

        # RandomForest original (usando todas las variables)
        X_rf = X
        y_rf = y
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X_rf, y_rf, test_size=0.3, stratify=y_rf, random_state=42
        )
        rf_full = RandomForestClassifier(n_estimators=500, random_state=42)
        rf_full.fit(X_train_rf, y_train_rf)
        y_pred_rf = rf_full.predict_proba(X_test_rf)[:, 1]

        fpr_rf, tpr_rf, _ = roc_curve(y_test_rf, y_pred_rf)
        auc_rf = auc(fpr_rf, tpr_rf)

        # ----------------------------
        # Curva ROC comparativa
        # ----------------------------
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines",
                                     name=f"RandomForest (Todas las vars, AUC={auc_rf:.3f})"))
        fig_roc.add_trace(go.Scatter(x=fpr_pca, y=tpr_pca, mode="lines",
                                     name=f"PCA (Componentes, AUC={auc_pca:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="gray"), showlegend=False))
        fig_roc.update_layout(title="ROC: PCA vs RandomForest",
                              xaxis_title="1 - Especificidad (FPR)",
                              yaxis_title="Sensibilidad (TPR)",
                              template="plotly_white")
        st.plotly_chart(fig_roc, use_container_width=True)

        # ----------------------------
        # Resumen y ganador
        # ----------------------------
        ganador_final = "RandomForest" if auc_rf >= auc_pca else "PCA"
        st.write(f"**AUC PCA:** {auc_pca:.3f}")
        st.write(f"**AUC RandomForest:** {auc_rf:.3f}")
        st.success(f"🏆 Método ganador según AUC: **{ganador_final}**")

        


