import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Configuración de página
st.set_page_config(
    page_title="Credit Risk Framework - Streamlit",
    page_icon="📊",
    layout="wide"
)

# Título Principal
st.title("🛡️ Credit Risk Framework - Prediction & Analytics")
st.markdown("Recreación interactiva en Streamlit de la solución predictiva para la evaluación de riesgo de impago (*Loan Default*).")

# Variables y estadísticas del proyecto original
means = {
    'Age': 43.498306,
    'Income': 82499.304597,
    'LoanAmount': 127578.865512,
    'CreditScore': 574.264346,
    'MonthsEmployed': 59.541976,
    'NumCreditLines': 2.501036,
    'InterestRate': 13.492773,
    'DTIRatio': 0.500212
}

stds = {
    'Age': 14.990258,
    'Income': 38963.013729,
    'LoanAmount': 70840.706142,
    'CreditScore': 158.903867,
    'MonthsEmployed': 34.643376,
    'NumCreditLines': 1.117018,
    'InterestRate': 6.636443,
    'DTIRatio': 0.230917
}

coefs = {
    'Age': -0.583122,
    'InterestRate': 0.458580,
    'MonthsEmployed': -0.337303,
    'Income': -0.313923,
    'LoanAmount': 0.288246,
    'NumCreditLines': 0.102686,
    'DTIRatio': 0.068616,
    'CreditScore': -0.125108,
    
    # Binarias
    'HasCoSigner': -0.258236,
    'HasDependents': -0.251843,
    'HasMortgage': -0.164002,
    
    # Categorical variables (One-hot encoded)
    'EmploymentType_Unemployed': 0.199126,
    'EmploymentType_Part-time': 0.029855,
    'EmploymentType_Self-employed': 0.008242,
    'EmploymentType_Full-time': -0.239942,
    
    'Education_High School': 0.133711,
    'Education_Bachelor\'s': 0.058102,
    'Education_Master\'s': -0.077604,
    'Education_PhD': -0.116928,
    
    'MaritalStatus_Divorced': 0.082256,
    'MaritalStatus_Single': 0.030815,
    'MaritalStatus_Married': -0.115791,
    
    'LoanPurpose_Business': 0.091138,
    'LoanPurpose_Auto': 0.027096,
    'LoanPurpose_Other': 0.018906,
    'LoanPurpose_Education': 0.017447,
    'LoanPurpose_Home': -0.157306
}

intercept = -0.001280

# Tabla de Métricas por Umbral obtenidas en el proyecto
thresholds_data = {
    'Threshold': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    'Accuracy': [0.143587, 0.194341, 0.258782, 0.328549, 0.401664, 0.474819, 0.547171, 0.614823, 0.676346, 0.731565, 0.777834, 0.815528, 0.845212, 0.867711, 0.880145, 0.884825, 0.884590],
    'Precision': [0.118999, 0.124888, 0.132655, 0.141955, 0.152940, 0.165653, 0.181397, 0.200026, 0.219541, 0.245218, 0.272192, 0.302333, 0.339512, 0.392270, 0.453523, 0.530973, 0.635036],
    'Recall': [0.995448, 0.988366, 0.971843, 0.947901, 0.914854, 0.872534, 0.825325, 0.772382, 0.699376, 0.631091, 0.545439, 0.450008, 0.352049, 0.253246, 0.156297, 0.070814, 0.014669]
}
df_thresholds = pd.DataFrame(thresholds_data)
df_thresholds['F1-Score'] = 2 * (df_thresholds['Precision'] * df_thresholds['Recall']) / (df_thresholds['Precision'] + df_thresholds['Recall'])

# Menú de Navegación Lateral
menu = st.sidebar.radio(
    "Navegación",
    ["Calculadora de Inferencia", "Análisis de Umbrales", "Visualizaciones de EDA", "Detalles del Modelo"]
)

if menu == "Calculadora de Inferencia":
    st.header("📝 Evaluación de Nuevo Solicitante")
    st.markdown("Ingresa los datos del cliente para calcular su probabilidad de impago en tiempo real utilizando el pipeline balanceado de regresión logística del proyecto.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=23)
        income = st.number_input("Ingreso Anual (USD)", min_value=0, value=40000)
        loan_amount = st.number_input("Monto del Préstamo (USD)", min_value=0, value=15000)
        credit_score = st.slider("Puntaje de Crédito (Credit Score)", 300, 850, 400)

    with col2:
        months_employed = st.number_input("Meses Empleado", min_value=0, value=24)
        num_credit_lines = st.slider("Líneas de Crédito Activas", 1, 15, 4)
        interest_rate = st.number_input("Tasa de Interés (%)", min_value=0.0, max_value=50.0, value=14.0)
        dti_ratio = st.slider("Relación Deuda-Ingreso (DTI Ratio)", 0.0, 1.0, 0.48)

    with col3:
        education = st.selectbox("Nivel Educativo", ["High School", "Bachelor's", "Master's", "PhD"])
        employment_type = st.selectbox("Tipo de Empleo", ["Unemployed", "Part-time", "Self-employed", "Full-time"])
        marital_status = st.selectbox("Estado Civil", ["Single", "Married", "Divorced"])
        loan_purpose = st.selectbox("Propósito del Préstamo", ["Business", "Auto", "Other", "Education", "Home"])
        
        has_mortgage = st.selectbox("¿Tiene Hipoteca?", ["No", "Yes"])
        has_dependents = st.selectbox("¿Tiene Dependientes?", ["No", "Yes"])
        has_cosigner = st.selectbox("¿Tiene Co-firmante / Aval?", ["No", "Yes"])

    # Selección de Umbral de decisión
    st.markdown("---")
    decision_threshold = st.slider("Umbral de Decisión de Riesgo", 0.05, 0.95, 0.50, 0.05,
                                  help="Las probabilidades mayores o iguales a este umbral clasificarán al solicitante como riesgo de DEFAULT.")

    # Cálculo del Score (Log-odds)
    z = intercept
    
    # Numéricas
    z += ((age - means['Age']) / stds['Age']) * coefs['Age']
    z += ((income - means['Income']) / stds['Income']) * coefs['Income']
    z += ((loan_amount - means['LoanAmount']) / stds['LoanAmount']) * coefs['LoanAmount']
    z += ((credit_score - means['CreditScore']) / stds['CreditScore']) * coefs['CreditScore']
    z += ((months_employed - means['MonthsEmployed']) / stds['MonthsEmployed']) * coefs['MonthsEmployed']
    z += ((num_credit_lines - means['NumCreditLines']) / stds['NumCreditLines']) * coefs['NumCreditLines']
    z += ((interest_rate - means['InterestRate']) / stds['InterestRate']) * coefs['InterestRate']
    z += ((dti_ratio - means['DTIRatio']) / stds['DTIRatio']) * coefs['DTIRatio']

    # Binarias
    z += (1.0 if has_mortgage == "Yes" else 0.0) * coefs['HasMortgage']
    z += (1.0 if has_dependents == "Yes" else 0.0) * coefs['HasDependents']
    z += (1.0 if has_cosigner == "Yes" else 0.0) * coefs['HasCoSigner']

    # Categóricas
    z += coefs.get(f"Education_{education}", 0.0)
    z += coefs.get(f"EmploymentType_{employment_type}", 0.0)
    z += coefs.get(f"MaritalStatus_{marital_status}", 0.0)
    z += coefs.get(f"LoanPurpose_{loan_purpose}", 0.0)

    # Probabilidad (función logística)
    prob_default = 1.0 / (1.0 + math.exp(-z))
    is_default = 1 if prob_default >= decision_threshold else 0

    st.subheader("📊 Resultado del Análisis")
    
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Probabilidad de Impago (Default)", f"{prob_default * 100:.2f}%")
        
        if is_default == 1:
            st.error("🚨 CLASIFICACIÓN: RIESGO ALTO (DEFAULT)")
        else:
            st.success("✅ CLASIFICACIÓN: RIESGO BAJO (APROBADO)")

    with res_col2:
        # Medidor de riesgo visual
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([0], [1.0], color="lightgray", height=0.4)
        ax.barh([0], [prob_default], color="red" if is_default == 1 else "green", height=0.4)
        ax.axvline(decision_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Umbral ({decision_threshold})")
        ax.set_xlim(0, 1.0)
        ax.set_yticks([])
        ax.set_xlabel("Probabilidad")
        ax.legend(loc="upper right", frameon=False)
        sns.despine(left=True, bottom=False, ax=ax)
        st.pyplot(fig)

    # Recomendación e Impacto de las Variables
    st.subheader("📋 Reporte de Decisiones e Influencias")
    contribs = [
        ("Edad", ((age - means['Age']) / stds['Age']) * coefs['Age']),
        ("Ingresos Anuales", ((income - means['Income']) / stds['Income']) * coefs['Income']),
        ("Monto de Préstamo", ((loan_amount - means['LoanAmount']) / stds['LoanAmount']) * coefs['LoanAmount']),
        ("Puntaje de Crédito", ((credit_score - means['CreditScore']) / stds['CreditScore']) * coefs['CreditScore']),
        ("Estabilidad Laboral (meses)", ((months_employed - means['MonthsEmployed']) / stds['MonthsEmployed']) * coefs['MonthsEmployed']),
        ("Líneas de Crédito", ((num_credit_lines - means['NumCreditLines']) / stds['NumCreditLines']) * coefs['NumCreditLines']),
        ("Tasa de Interés", ((interest_rate - means['InterestRate']) / stds['InterestRate']) * coefs['InterestRate']),
        ("Índice DTI", ((dti_ratio - means['DTIRatio']) / stds['DTIRatio']) * coefs['DTIRatio']),
        ("Aval/Codeudor", (1.0 if has_cosigner == "Yes" else 0.0) * coefs['HasCoSigner']),
        ("Hipoteca", (1.0 if has_mortgage == "Yes" else 0.0) * coefs['HasMortgage']),
        ("Nivel Educativo", coefs.get(f"Education_{education}", 0.0)),
        ("Tipo de Empleo", coefs.get(f"EmploymentType_{employment_type}", 0.0)),
        ("Propósito Préstamo", coefs.get(f"LoanPurpose_{loan_purpose}", 0.0)),
    ]
    
    df_contrib = pd.DataFrame(contribs, columns=["Factor", "Influencia en el Riesgo (Log-Odds)"])
    df_contrib["Dirección"] = df_contrib["Influencia en el Riesgo (Log-Odds)"].apply(lambda x: "Aumenta Riesgo 🔺" if x > 0 else "Disminuye Riesgo 🟢" if x < 0 else "Neutral ⚪")
    df_contrib["Magnitud Absoluta"] = df_contrib["Influencia en el Riesgo (Log-Odds)"].abs()
    df_contrib = df_contrib.sort_values(by="Magnitud Absoluta", ascending=False).drop(columns=["Magnitud Absoluta"])

    st.dataframe(df_contrib, use_container_width=True)

    if is_default == 1:
        st.markdown("**Recomendación de Negocio:** El perfil de este cliente supera el nivel de riesgo tolerado. "
                    "Se sugiere una **revisión manual**, solicitar un **aval/codeudor**, reducir el **monto del préstamo**, "
                    "o ajustar la **tasa de interés** para mitigar pérdidas.")
    else:
        st.markdown("**Recomendación de Negocio:** El perfil del cliente se mantiene dentro de los límites de riesgo admisibles. "
                    "**Aprobación recomendada** bajo los términos actuales del préstamo.")

elif menu == "Análisis de Umbrales":
    st.header("📈 Simulador de Umbral y Negociación de Errores")
    st.markdown("Ajusta el umbral de decisión para observar el intercambio clásico entre **Precision** y **Recall**, "
                "y evalúa el impacto financiero directo que esto causa sobre un banco ficticio de 51,070 clientes probados.")

    threshold_sel = st.slider("Seleccionar Umbral", 0.10, 0.90, 0.50, 0.05)

    # Gráfico de curvas de rendimiento
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_thresholds['Threshold'], df_thresholds['Precision'], label='Precision', marker='o')
    ax.plot(df_thresholds['Threshold'], df_thresholds['Recall'], label='Recall (Exhaustividad)', marker='s')
    ax.plot(df_thresholds['Threshold'], df_thresholds['F1-Score'], label='F1-Score', linestyle='--', color='gray')
    ax.axvline(threshold_sel, color='red', linestyle=':', linewidth=2, label=f'Umbral Seleccionado ({threshold_sel})')
    ax.set_xlabel('Umbral de Decisión')
    ax.set_ylabel('Score')
    ax.set_title('Métricas según el Umbral (Regresión Logística Balanceada)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Obtención de métricas para el umbral seleccionado
    row_metrics = df_thresholds[np.isclose(df_thresholds['Threshold'], threshold_sel)].iloc[0]
    
    st.subheader(f"📊 Desempeño del Modelo en Umbral {threshold_sel}")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Recall (Defaults Detectados)", f"{row_metrics['Recall'] * 100:.2f}%")
    m_col2.metric("Precision (Precisión en Alertas)", f"{row_metrics['Precision'] * 100:.2f}%")
    m_col3.metric("F1-Score", f"{row_metrics['F1-Score']:.4f}")
    m_col4.metric("Exactitud General (Accuracy)", f"{row_metrics['Accuracy'] * 100:.2f}%")

    # Matriz de Confusión Simulada (Sobre 51,070 clientes del test set)
    # Total de defaults reales en test set = 5,931. No defaults reales = 45,139.
    total_default = 5931
    total_no_default = 45139
    
    tp = int(round(row_metrics['Recall'] * total_default))
    fn = total_default - tp
    # Precision = tp / (tp + fp)  --> fp = tp / precision - tp
    fp = int(round(tp / row_metrics['Precision'] - tp)) if row_metrics['Precision'] > 0 else 0
    tn = total_no_default - fp

    st.subheader("🔲 Matriz de Confusión Simulada (Conjunto de Prueba)")
    conf_matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Real: Paga (0)", "Real: Default (1)"],
        columns=["Predicción: Paga (0)", "Predicción: Default (1)"]
    )
    st.table(conf_matrix)

    # Análisis de impacto financiero
    st.subheader("💰 Simulación de Impacto Financiero en el Negocio")
    st.markdown("Asignemos un costo financiero simplificado para entender la toma de decisiones:")
    st.markdown("- **Falso Negativo (Error Grave):** El banco aprueba un préstamo que cae en default. Costo promedio: **$10,000 USD**.")
    st.markdown("- **Falso Positivo (Inconveniente):** El banco frena/revisa a un cliente que sí iba a pagar. Costo por revisión/pérdida: **$500 USD**.")

    fn_cost = fn * 10000
    fp_cost = fp * 500
    total_cost = fn_cost + fp_cost

    st.error(f"Pérdida por Defaults no detectados (FN): **${fn_cost:,.2f} USD** ({fn} clientes)")
    st.warning(f"Costo por fricción/revisiones extras (FP): **${fp_cost:,.2f} USD** ({fp} revisiones)")
    st.success(f"**Costo Total de Decisiones Equivocadas:** **${total_cost:,.2f} USD**")

elif menu == "Visualizaciones de EDA":
    st.header("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("Visualización de las relaciones clave descubiertas durante el desarrollo del proyecto financiero.")

    eda_option = st.selectbox("Selecciona un gráfico de EDA", [
        "Capacidad Financiera: Ingreso vs Monto Préstamo",
        "Estabilidad Laboral vs Tipo de Empleo",
        "Distribución de Edades por Default"
    ])

    if eda_option == "Capacidad Financiera: Ingreso vs Monto Préstamo":
        st.markdown("**Bloque 1: Capacidad Financiera.** En este mapa de calor simulado del proyecto, se observa que la tasa más alta de default (30.51%) se ubica en el cuadrante de bajos ingresos y montos solicitados altos.")
        
        # Recreación del Heatmap del reporte original (Figura 5)
        income_bands = ['($15k, $42k]', '($42k, $69k]', '($69k, $95k]', '($95k, $122k]', '($122k, $150k]']
        loan_bands = ['($5k, $53k]', '($53k, $102k]', '($102k, $152k]', '($152k, $201k]', '($201k, $250k]']
        
        data_heatmap = np.array([
            [9.02, 12.98, 17.27, 22.94, 30.51],
            [7.75, 9.64, 10.80, 13.27, 15.37],
            [7.68, 8.90, 9.77, 10.85, 12.13],
            [8.20, 8.30, 9.21, 10.16, 10.68],
            [7.60, 8.50, 8.86, 9.43, 10.48]
        ])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data_heatmap, xticklabels=loan_bands, yticklabels=income_bands, 
                    annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Tasa de Default (%)'}, ax=ax)
        ax.set_title("Tasa de Default por Ingreso y Monto de Préstamo")
        ax.set_xlabel("Monto del Préstamo")
        ax.set_ylabel("Ingresos Anuales del Prestatario")
        st.pyplot(fig)

    elif eda_option == "Estabilidad Laboral vs Tipo de Empleo":
        st.markdown("**Bloque 3: Estabilidad laboral.** La antigüedad laboral (MonthsEmployed) tiene mayor peso directo que el tipo de empleo (EmploymentType). La tasa cae progresivamente al aumentar la experiencia laboral en todos los sectores.")
        
        experience_bands = ['(0, 24m]', '(24m, 48m]', '(48m, 72m]', '(72m, 96m]', '(96m, 119m]']
        employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
        
        data_emp = np.array([
            [13.39, 17.16, 15.75, 18.67],
            [11.07, 13.89, 13.63, 15.47],
            [8.83, 11.31, 10.88, 13.21],
            [7.57, 9.35, 8.96, 11.30],
            [6.14, 7.80, 7.75, 8.75]
        ])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data_emp, xticklabels=employment_types, yticklabels=experience_bands,
                    annot=True, fmt=".2f", cmap="mako", cbar_kws={'label': 'Tasa de Default (%)'}, ax=ax)
        ax.set_title("Tasa de Default por Tipo de Empleo y Antigüedad (Meses)")
        ax.set_xlabel("Tipo de Empleo")
        ax.set_ylabel("Meses Empleado")
        st.pyplot(fig)

    elif eda_option == "Distribución de Edades por Default":
        st.markdown("**Bloque 4: Perfil del solicitante.** El incumplimiento se concentra significativamente en clientes más jóvenes (media cercana a 35 años), reduciéndose el riesgo a medida que incrementa la edad.")
        
        np.random.seed(42)
        age_no_default = np.random.normal(45, 12, 1000)
        age_default = np.random.normal(35, 10, 1000)
        
        # Filtro de límites reales de edad del dataset (18 a 69 años)
        age_no_default = age_no_default[(age_no_default >= 18) & (age_no_default <= 69)]
        age_default = age_default[(age_default >= 18) & (age_default <= 69)]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(age_no_default, fill=True, label="Paga (0)", ax=ax, color="blue", alpha=0.3)
        sns.kdeplot(age_default, fill=True, label="Default (1)", ax=ax, color="orange", alpha=0.5)
        ax.set_title("Densidad de Distribución de Edad según Default")
        ax.set_xlabel("Edad")
        ax.set_ylabel("Densidad")
        ax.legend()
        st.pyplot(fig)

elif menu == "Detalles del Modelo":
    st.header("🧠 Coeficientes y Factores Más Influyentes")
    st.markdown("La Regresión Logística balanceada es altamente interpretable gracias a sus coeficientes. "
                "Los coeficientes positivos aumentan la probabilidad de impago, mientras que los negativos actúan como mitigantes de riesgo.")

    # Recreación de coeficientes ordenados por valor absoluto (Figura 25)
    coefs_plot = {
        'Edad (Age)': -0.583122,
        'Tasa de Interés (InterestRate)': 0.458580,
        'Meses de Empleo (MonthsEmployed)': -0.337303,
        'Ingresos (Income)': -0.313923,
        'Monto del Préstamo (LoanAmount)': 0.288246,
        'Avalista / Co-firmante (HasCoSigner)': -0.258236,
        'Dependientes (HasDependents)': -0.251843,
        'Empleo: Tiempo Completo (Full-time)': -0.239942,
        'Empleo: Desempleado (Unemployed)': 0.199126,
        'Hipoteca (HasMortgage)': -0.164002,
        'Propósito: Hogar (Home)': -0.157306,
        'Educación: Preparatoria (High School)': 0.133711,
        'Puntaje de Crédito (CreditScore)': -0.125108,
        'Educación: PhD': -0.116928,
        'Estado Civil: Casado (Married)': -0.115791
    }
    
    df_coefs = pd.DataFrame(list(coefs_plot.items()), columns=['Variable', 'Coeficiente'])
    df_coefs['Abs_Coef'] = df_coefs['Coeficiente'].abs()
    df_coefs = df_coefs.sort_values(by='Abs_Coef', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['firebrick' if val > 0 else 'forestgreen' for val in df_coefs['Coeficiente']]
    bars = ax.barh(df_coefs['Variable'], df_coefs['Coeficiente'], color=colors, height=0.6)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_title("Coeficientes de Regresión Logística (Impacto Directo)")
    ax.set_xlabel("Valor del Coeficiente (Log-Odds)")
    
    # Agregar etiquetas con los valores en los extremos de las barras
    for bar, val in zip(bars, df_coefs['Coeficiente']):
        width = bar.get_width()
        ax.text(width + (0.01 if val >= 0 else -0.06), bar.get_y() + bar.get_height()/2, 
                f"{val:+.3f}", 
                va='center', ha='left' if val >= 0 else 'right', fontsize=9, fontweight='bold')

    sns.despine(ax=ax)
    st.pyplot(fig)
    st.info("💡 Consejo: Los coeficientes negativos (verdes) reducen el riesgo de impago, "
            "mientras que los coeficientes positivos (rojos) incrementan el riesgo crediticio.")
