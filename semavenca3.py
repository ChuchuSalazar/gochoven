import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import fsolve
import math

# Configuración de página
st.set_page_config(
    page_title="SEMAVENCA - Análisis Financiero",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header con copyright
st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79, #2e86ab); padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h1 style="color: white; text-align: center; margin: 0;">📊 SEMAVENCA</h1>
        <p style="color: white; text-align: center; margin: 5px 0 0 0; font-size: 14px;">© 2024 SEMAVENCA - Alcaldía Munic. Sucre Edo. Miranda</p>
    </div>
""", unsafe_allow_html=True)


def calcular_tir(flujos_caja):
    """Calcular TIR usando método numérico"""
    def van_eq(tasa):
        try:
            return sum([flujo / (1 + tasa)**i for i, flujo in enumerate(flujos_caja)])
        except:
            return float('inf')

    try:
        tir = fsolve(van_eq, 0.1)[0]
        return tir if abs(van_eq(tir)) < 1e-6 and tir > -0.99 else None
    except:
        return None


def calcular_van(flujos_caja, tasa_descuento):
    """Calcular VAN"""
    try:
        return sum([flujo / (1 + tasa_descuento)**i for i, flujo in enumerate(flujos_caja)])
    except:
        return 0


# Sidebar para inputs
st.sidebar.header("⚙️ Parámetros de Inversión")

# Botón de reinicio
if st.sidebar.button("🔄 Reiniciar Valores", type="secondary", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")

# Inputs principales
inversion_inicial = st.sidebar.number_input(
    "💰 Inversión Inicial (II)",
    min_value=0.0,
    value=2482400.0,
    step=10000.0,
    format="%.2f",
    key="inversion"
)

ingreso_mensual_inicial = st.sidebar.number_input(
    "📈 Ingreso Mensual Inicial (IM)",
    min_value=0.0,
    value=415872.0,
    step=1000.0,
    format="%.2f",
    key="ingreso"
)

# Horizonte del proyecto con slider
st.sidebar.markdown("### 📅 Horizonte del Proyecto")
horizonte_meses = st.sidebar.slider(
    "Duración del proyecto (meses)",
    min_value=12,
    max_value=120,
    value=48,
    step=6,
    key="horizonte",
    help="Seleccione la duración total del proyecto"
)

# Mostrar línea de tiempo visual
st.sidebar.markdown(f"""
<div style="background: linear-gradient(90deg, #e3f2fd 0%, #1976d2 {(48/horizonte_meses)*100}%, #f5f5f5 100%); 
            height: 20px; border-radius: 10px; margin: 10px 0;">
</div>
<div style="display: flex; justify-content: space-between; font-size: 12px;">
    <span>Mes 0</span>
    <span>Mes {horizonte_meses//2}</span>
    <span>Mes {horizonte_meses}</span>
</div>
""", unsafe_allow_html=True)

tasa_variacion = st.sidebar.number_input(
    "📊 Tasa de Variación Intermensual (%)",
    min_value=-10.0,
    max_value=20.0,
    value=2.0,
    step=0.1,
    format="%.2f",
    key="tasa_var"
) / 100

# Generar ingresos mensuales para el horizonte seleccionado
ingresos_mensuales = []
for mes in range(1, horizonte_meses + 1):
    if mes <= 3:  # Primeros 3 meses sin ingresos
        ingresos_mensuales.append(0)
    else:  # A partir del mes 4
        ingreso = ingreso_mensual_inicial * (1 + tasa_variacion) ** (mes - 4)
        ingresos_mensuales.append(ingreso)

# Construir flujos de caja para cálculo de TIR
flujos_caja = [-inversion_inicial] + ingresos_mensuales

# Calcular TIR con el nuevo horizonte
tir_calculada = calcular_tir(flujos_caja)

# Mostrar la TIR actual calculada
st.sidebar.info(
    f"TIR Calculada: {tir_calculada*100:.2f}%" if tir_calculada else "TIR: No calculable"
)

# Tasa de descuento
tasa_descuento = st.sidebar.number_input(
    "🎯 Tasa de Interés/Descuento (%)",
    min_value=0.0,
    max_value=50.0,
    value=tir_calculada * 100 if tir_calculada else 10.0,
    step=0.1,
    format="%.2f",
    key="tasa_desc",
    help=f"TIR actual: {tir_calculada*100:.2f}%" if tir_calculada else "TIR no calculable"
) / 100

# Calcular métricas
ingreso_acumulado = [0] + \
    list(np.cumsum(ingresos_mensuales))  # Mes 0 + acumulados
monto_amortizado = [x - inversion_inicial for x in ingreso_acumulado]
van = calcular_van(flujos_caja, tasa_descuento)

# Encontrar mes de recuperación
mes_recuperacion = None
for i, monto in enumerate(monto_amortizado):
    if monto >= 0:
        mes_recuperacion = i
        break

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Análisis de Flujo de Caja")

    # Crear DataFrame para la tabla
    meses_array = list(range(len(flujos_caja)))
    ingresos_array = [0] + ingresos_mensuales  # Mes 0 sin ingresos

    df_resultados = pd.DataFrame({
        'Mes': meses_array,
        'Ingreso Mensual': ingresos_array,
        'Ingreso Acumulado': ingreso_acumulado,
        'Monto Amortizado': monto_amortizado,
        'Flujo de Caja': flujos_caja
    })

    # Crear copia para valores numéricos
    df_numeric = df_resultados.copy()

    # Formatear valores monetarios para mostrar
    for col in ['Ingreso Mensual', 'Ingreso Acumulado', 'Monto Amortizado', 'Flujo de Caja']:
        df_resultados[col] = df_resultados[col].apply(lambda x: f"${x:,.2f}")

    # Mostrar tabla (primeros 20 meses o todos si es menor)
    filas_mostrar = min(21, horizonte_meses + 1)
    st.write(f"**Tabla de Flujos (Primeros {filas_mostrar-1} meses):**")
    st.dataframe(df_resultados.head(filas_mostrar), use_container_width=True)

with col2:
    st.subheader("🎯 Métricas Clave")

    # Mostrar TIR calculada
    if tir_calculada:
        st.metric("TIR", f"{tir_calculada*100:.2f}%",
                  delta="Se actualiza con el horizonte")
    else:
        st.metric("TIR", "No calculable")

    # Métricas en cards
    st.metric("VAN", f"${van:,.2f}",
              delta="Positivo" if van > 0 else "Negativo")

    st.metric("Tasa Descuento Actual", f"{tasa_descuento*100:.2f}%",
              delta=f"TIR: {tir_calculada*100:.2f}%" if tir_calculada else None)

    st.metric("Mes de Recuperación",
              f"Mes {mes_recuperacion}" if mes_recuperacion else f"No se recupera en {horizonte_meses} meses")

    st.metric("Inversión Inicial", f"${inversion_inicial:,.2f}")

    if ingreso_acumulado[-1] > 0:
        st.metric(
            "ROI Total", f"{((ingreso_acumulado[-1]/inversion_inicial)-1)*100:.1f}%")
    else:
        st.metric("ROI Total", "0.0%")

# Gráficos
st.subheader("📈 Visualizaciones")

# Crear subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Evolución del Monto Amortizado', 'Ingresos Mensuales',
                    'Ingreso Acumulado vs Inversión', 'Flujo de Caja Mensual'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

meses = list(range(len(df_resultados)))

# Gráfico 1: Monto Amortizado
fig.add_trace(
    go.Scatter(x=meses, y=df_numeric['Monto Amortizado'], name="Monto Amortizado",
               line=dict(color='red', width=3)),
    row=1, col=1
)
fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

# Gráfico 2: Ingresos Mensuales
fig.add_trace(
    go.Bar(x=meses[1:], y=df_numeric['Ingreso Mensual'][1:], name="Ingresos Mensuales",
           marker_color='lightblue'),
    row=1, col=2
)

# Gráfico 3: Ingreso Acumulado
fig.add_trace(
    go.Scatter(x=meses, y=df_numeric['Ingreso Acumulado'], name="Ingreso Acumulado",
               line=dict(color='green', width=3)),
    row=2, col=1
)
fig.add_hline(y=inversion_inicial, line_dash="dash", line_color="red",
              annotation_text="Inversión Inicial", row=2, col=1)

# Gráfico 4: Flujo de Caja
colors = ['red' if x < 0 else 'green' for x in df_numeric['Flujo de Caja']]
fig.add_trace(
    go.Bar(x=meses, y=df_numeric['Flujo de Caja'], name="Flujo de Caja",
           marker_color=colors),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False,
                  title_text="Dashboard de Análisis Financiero")
fig.update_xaxes(title_text="Meses")
fig.update_yaxes(title_text="Monto ($)")

st.plotly_chart(fig, use_container_width=True)

# Análisis y recomendaciones
st.subheader("📋 Análisis y Recomendaciones")

col1, col2, col3 = st.columns(3)

with col1:
    if van > 0:
        st.success("✅ **VAN Positivo**: El proyecto genera valor")
    else:
        st.error("❌ **VAN Negativo**: El proyecto destruye valor")

with col2:
    if tir_calculada and tir_calculada > tasa_descuento:
        st.success(
            f"✅ **TIR Superior**: {tir_calculada*100:.2f}% > {tasa_descuento*100:.2f}%")
    elif tir_calculada:
        st.error(
            f"❌ **TIR Inferior**: {tir_calculada*100:.2f}% < {tasa_descuento*100:.2f}%")
    else:
        st.error("❌ **TIR No Calculable**")

with col3:
    if mes_recuperacion and mes_recuperacion <= 24:
        st.success(f"✅ **Recuperación Rápida**: {mes_recuperacion} meses")
    elif mes_recuperacion:
        st.warning(f"⚠️ **Recuperación Lenta**: {mes_recuperacion} meses")
    else:
        st.error(
            f"❌ **Sin Recuperación** en el horizonte de {horizonte_meses} meses")

# Exportar datos
st.subheader("💾 Exportar Resultados")

try:
    # Preparar datos para Excel
    df_export = df_numeric.copy()  # Usar valores numéricos

    # Agregar hoja de resumen
    resumen_data = {
        'Métrica': ['Inversión Inicial', 'Horizonte (meses)', 'TIR (%)', 'Tasa de Descuento (%)', 'VAN', 'Mes de Recuperación', 'ROI Total (%)'],
        'Valor': [
            inversion_inicial,
            horizonte_meses,
            tir_calculada * 100 if tir_calculada else 0,
            tasa_descuento * 100,
            van,
            mes_recuperacion if mes_recuperacion else 'No se recupera',
            ((ingreso_acumulado[-1]/inversion_inicial)-1) *
            100 if ingreso_acumulado[-1] > 0 else 0
        ]
    }
    df_resumen = pd.DataFrame(resumen_data)

    # Crear archivo Excel con múltiples hojas
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='Flujos_Detallados', index=False)
        df_resumen.to_excel(writer, sheet_name='Resumen_Metricas', index=False)

    st.download_button(
        label="📥 Descargar Análisis Excel",
        data=output.getvalue(),
        file_name=f'analisis_financiero_semavenca_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

except ImportError:
    st.warning("⚠️ Para exportar a Excel, instale la dependencia 'openpyxl'")

    # Opción alternativa: exportar como CSV
    csv_data = df_numeric.to_csv(index=False)
    st.download_button(
        label="📥 Descargar como CSV",
        data=csv_data,
        file_name=f'analisis_financiero_semavenca_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Desarrollado por SEMAVENCA/JFSR © 2024</div>",
    unsafe_allow_html=True
)
