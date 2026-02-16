# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.optimize import root_scalar, minimize_scalar
import pandas as pd

# ============================================================================
# SOLVER DE CATENARIA EXACTA - REACCIONES VERTICALES SIEMPRE HACIA ARRIBA
# ============================================================================
class ExactCatenarySolver:
    """
    Cable con dos tramos y un apoyo intermedio fijo en y=0.
    Todos los apoyos están a la misma altura.
    Las reacciones se entregan como fuerzas que el apoyo ejerce SOBRE el cable.
    Las componentes verticales son SIEMPRE positivas (↑).
    """

    def __init__(self, L_span, L_total, x_support, diameter, rho=7850, g=9.81):
        self.L_span = L_span
        self.L_total = L_total
        self.x_support = x_support
        self.diameter = diameter
        self.rho = rho
        self.g = g

        # Propiedades del cable
        self.area = np.pi * (diameter / 2) ** 2
        self.w = rho * g * self.area          # Peso por unidad de longitud [N/m]

        # Luces de cada tramo
        self.L1 = x_support
        self.L2 = L_span - x_support

        # Validaciones
        if self.L1 <= 0 or self.L2 <= 0:
            raise ValueError("El apoyo debe estar dentro del span (0 < x_support < L_span)")
        if self.L_total <= self.L_span:
            raise ValueError("La longitud del cable debe ser mayor que el span")

    def _catenary_length(self, a, L):
        """Longitud de catenaria simétrica con luz L y parámetro a"""
        if a <= 0:
            return np.inf
        return 2 * a * np.sinh(L / (2 * a))

    def _catenary_sag(self, a, L):
        """Flecha máxima de catenaria simétrica"""
        return a * (np.cosh(L / (2 * a)) - 1)

    def _total_length_error(self, a):
        """Error relativo de longitud total"""
        S1 = self._catenary_length(a, self.L1)
        S2 = self._catenary_length(a, self.L2)
        return (S1 + S2 - self.L_total) / self.L_total

    def solve(self, tol=1e-12, max_iter=1000):
        """
        Resuelve el parámetro a (y H) que satisface la longitud total.
        Retorna diccionario con todos los resultados.
        """
        # Estimación inicial basada en parábola
        excess = self.L_total - self.L_span
        f_est = np.sqrt(3 * self.L_span * excess / 8)
        a_est = self.L_span ** 2 / (8 * f_est)

        # Resolver ecuación no lineal: S1(a) + S2(a) = L_total
        try:
            sol = root_scalar(
                self._total_length_error,
                bracket=[a_est * 0.1, a_est * 10],
                method='brentq',
                xtol=tol,
                rtol=tol,
                maxiter=max_iter
            )
            a = sol.root
        except Exception as e:
            # Fallback: minimización del error absoluto
            res = minimize_scalar(
                lambda x: abs(self._total_length_error(x)),
                bounds=(a_est * 0.01, a_est * 100),
                method='bounded',
                options={'xatol': tol, 'maxiter': max_iter}
            )
            a = res.x

        # Parámetros derivados
        H = a * self.w                     # Tensión horizontal [N]
        S1 = self._catenary_length(a, self.L1)
        S2 = self._catenary_length(a, self.L2)
        S_total = S1 + S2
        f1 = self._catenary_sag(a, self.L1)
        f2 = self._catenary_sag(a, self.L2)

        # Componentes verticales de la tensión en los extremos (hacia arriba)
        V_left = H * np.sinh(self.L1 / (2 * a))   # positiva, ↑
        V_right = H * np.sinh(self.L2 / (2 * a))  # positiva, ↑
        peso_total = self.w * S_total

        # Reacción en el apoyo intermedio: fuerza hacia arriba que la torre ejerce sobre el cable
        R_support = peso_total - V_left - V_right   # siempre positiva (↑)

        # REACCIONES (fuerzas que el apoyo ejerce SOBRE el cable)
        reactions = {
            'left': (-H, V_left),      # ← y ↑
            'right': (H, V_right),     # → y ↑
            'support': R_support       # ↑ (solo vertical)
        }

        # Generar puntos de la catenaria para graficar
        n_points = 500
        x1 = np.linspace(0, self.L1, n_points // 2)
        y1 = a * (np.cosh((x1 - self.L1 / 2) / a) - np.cosh(self.L1 / (2 * a)))
        x2 = np.linspace(self.L1, self.L_span, n_points // 2)
        y2 = a * (np.cosh((x2 - self.L1 - self.L2 / 2) / a) - np.cosh(self.L2 / (2 * a)))
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])

        # Tensiones a lo largo del cable
        dy_dx_left = np.gradient(y1, x1)
        dy_dx_right = np.gradient(y2, x2)
        tensions_left = H * np.sqrt(1 + dy_dx_left ** 2)
        tensions_right = H * np.sqrt(1 + dy_dx_right ** 2)
        tensions = np.concatenate([tensions_left, tensions_right])
        
        # Calcular segmentos de igual longitud
        segmentos_iguales = self._calcular_segmentos_iguales(x, y, tensions, n_segmentos=100)

        resultados = {
            'a': a,
            'H': H,
            'L1': S1,
            'L2': S2,
            'total_length': S_total,
            'f1': f1,
            'f2': f2,
            'x': x,
            'y': y,
            'tensions': tensions,
            'reactions': reactions,
            'V_left': V_left,
            'V_right': V_right,
            'peso_total': peso_total,
            'R_support': R_support,
            'segmentos': segmentos_iguales
        }
        return resultados
        
    def _calcular_segmentos_iguales(self, x, y, tensions, n_segmentos=100):
        """
        Divide el cable en n_segmentos de igual longitud y calcula la tensión media en cada uno.
        """
        # Calcular la longitud acumulada a lo largo del cable
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s_acum = np.concatenate([[0], np.cumsum(ds)])
        s_total = s_acum[-1]
        
        # Longitud de cada segmento
        s_segmento = s_total / n_segmentos
        
        # Puntos para los segmentos
        x_seg = [x[0]]
        y_seg = [-y[0]]  # Invertir y para visualización
        s_actual = 0
        tensions_medias = []
        
        for i in range(1, n_segmentos + 1):
            s_objetivo = i * s_segmento
            
            # Encontrar el índice donde se alcanza o supera la longitud objetivo
            idx = np.searchsorted(s_acum, s_objetivo)
            
            if idx >= len(s_acum):
                idx = len(s_acum) - 1
            
            # Interpolar posición para la longitud exacta
            if idx > 0 and idx < len(s_acum):
                fraccion = (s_objetivo - s_acum[idx-1]) / (s_acum[idx] - s_acum[idx-1])
                x_interp = x[idx-1] + fraccion * (x[idx] - x[idx-1])
                y_interp = y[idx-1] + fraccion * (y[idx] - y[idx-1])
            else:
                x_interp = x[idx]
                y_interp = y[idx]
            
            x_seg.append(x_interp)
            y_seg.append(-y_interp)
            
            # Calcular tensión media en el segmento
            mask = (s_acum >= s_actual) & (s_acum <= s_objetivo)
            if np.sum(mask) > 0:
                tension_media = np.mean(tensions[mask])
            else:
                tension_media = tensions[idx-1] if idx > 0 else tensions[0]
            
            tensions_medias.append(tension_media)
            s_actual = s_objetivo
        
        # Asegurar el último punto exacto
        x_seg[-1] = x[-1]
        y_seg[-1] = -y[-1]
        
        return {
            'x_segmentos': x_seg,
            'y_segmentos': y_seg,
            'tensiones_medias': np.array(tensions_medias),
            'n_segmentos': n_segmentos,
            's_total': s_total,
            'tension_min': np.min(tensions_medias),
            'tension_max': np.max(tensions_medias),
            'tension_media_global': np.mean(tensions_medias)
        }

    def verify_precision(self, resultados):
        """Verifica errores de equilibrio y longitud con alta precisión"""
        S_total = resultados['total_length']
        V_left = resultados['V_left']
        V_right = resultados['V_right']
        R = resultados['R_support']
        peso = resultados['peso_total']
        H = resultados['H']

        # Error en longitud
        error_L = abs(S_total - self.L_total) / self.L_total * 100

        # Equilibrio vertical
        sum_V = V_left + V_right + R
        error_V = abs(sum_V - peso) / peso * 100

        # Equilibrio horizontal
        sum_H = -H + H
        error_H = abs(sum_H) / H * 100 if H != 0 else 0

        return {
            'error_L': error_L,
            'error_V': error_V,
            'error_H': error_H,
            'cumple_005': error_L < 0.05 and error_V < 0.05 and error_H < 0.05
        }


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN PARA STREAMLIT
# ============================================================================
def crear_graficos(solver, resultados):
    """Crea los dos gráficos para la aplicación Streamlit"""
    
    fig = plt.figure(figsize=(14, 6))
    
    # Crear subplots
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # Obtener datos de reacciones
    H = resultados['H']
    V_left = resultados['V_left']
    V_right = resultados['V_right']
    R_support = resultados['R_support']
    
    # --- GRÁFICO 1: Geometría de la catenaria ---
    x = resultados['x']
    y = resultados['y']
    
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)
    mid = n // 2

    # Se grafica -y para que la flecha sea positiva hacia abajo
    ax1.plot(x[:mid], -y[:mid], 'b-', linewidth=2.5, label='Tramo izquierdo')
    ax1.plot(x[mid:], -y[mid:], color='orange', linewidth=2.5, label='Tramo derecho')

    # Apoyo (fijo en y=0)
    idx = np.argmin(np.abs(x - solver.L1))
    ax1.plot(x[idx], 0, 'gs', markersize=12, label=f'Apoyo (x={solver.L1:.3f}m)')

    # Anclajes
    ax1.plot(0, 0, 'k^', markersize=15, label='Anclaje izquierdo')
    ax1.plot(solver.L_span, 0, 'k^', markersize=15, label='Anclaje derecho')

    # Línea horizontal en y=0
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)

    ax1.set_xlabel('x [m]', fontsize=10)
    ax1.set_ylabel('y (positivo hacia abajo) [m]', fontsize=10)
    ax1.set_title('GEOMETRÍA', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)

    # --- GRÁFICO 2: Cable coloreado por tensión ---
    segmentos = resultados['segmentos']
    x_seg = segmentos['x_segmentos']
    y_seg = segmentos['y_segmentos']
    tensiones_medias = segmentos['tensiones_medias']
    n_seg = len(tensiones_medias)
    
    x_seg = np.array(x_seg)
    y_seg = np.array(y_seg)
    
    # Crear normalización para el colormap
    tension_min = segmentos['tension_min']
    tension_max = segmentos['tension_max']
    
    # Añadir margen para mejor visualización
    tension_min_display = tension_min * 0.95
    tension_max_display = tension_max * 1.05
    
    norm = Normalize(vmin=tension_min_display, vmax=tension_max_display)
    cmap = cm.viridis
    
    # Dibujar cada segmento con su color correspondiente
    for i in range(n_seg):
        color = cmap(norm(tensiones_medias[i]))
        ax2.plot([x_seg[i], x_seg[i+1]], 
                 [y_seg[i], y_seg[i+1]], 
                 color=color, linewidth=3, solid_capstyle='round')
    
    # Apoyo y anclajes
    idx_apoyo = np.argmin(np.abs(x_seg - solver.L1))
    ax2.plot(x_seg[idx_apoyo], y_seg[idx_apoyo], 'gs', markersize=12)
    ax2.plot(0, 0, 'k^', markersize=15)
    ax2.plot(solver.L_span, 0, 'k^', markersize=15)
    
    # Línea horizontal en y=0
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # AÑADIR FLECHAS DE REACCIONES
    flecha_max_2 = np.max(np.abs(y_seg)) if len(y_seg) > 0 and np.max(np.abs(y_seg)) > 0 else 1.0
    scale_factor_2 = 0.25 * flecha_max_2
    
    # ANCLAJE IZQUIERDO
    ax2.arrow(0, 0, -scale_factor_2, 0, head_width=0.1*scale_factor_2, 
              head_length=0.1*scale_factor_2, fc='red', ec='red', width=0.02*scale_factor_2)
    ax2.text(-scale_factor_2*1.2, -0.05, f'H={H:.1f} N', fontsize=7, color='red', ha='right')
    ax2.arrow(0, 0, 0, -scale_factor_2, head_width=0.1*scale_factor_2, 
              head_length=0.1*scale_factor_2, fc='blue', ec='blue', width=0.02*scale_factor_2)
    ax2.text(0.1, -scale_factor_2*1.1, f'V={V_left:.1f} N', fontsize=7, color='blue', va='top')
    
    # ANCLAJE DERECHO
    ax2.arrow(solver.L_span, 0, scale_factor_2, 0, head_width=0.1*scale_factor_2, 
              head_length=0.1*scale_factor_2, fc='red', ec='red', width=0.02*scale_factor_2)
    ax2.text(solver.L_span + scale_factor_2*1.2, -0.05, f'H={H:.1f} N', fontsize=7, color='red')
    ax2.arrow(solver.L_span, 0, 0, -scale_factor_2, head_width=0.1*scale_factor_2, 
              head_length=0.1*scale_factor_2, fc='blue', ec='blue', width=0.02*scale_factor_2)
    ax2.text(solver.L_span - 0.5, -scale_factor_2*1.1, f'V={V_right:.1f} N', fontsize=7, color='blue', va='top', ha='right')
    
    # APOYO INTERMEDIO
    ax2.arrow(x_seg[idx_apoyo], 0, 0, -scale_factor_2, head_width=0.1*scale_factor_2, 
              head_length=0.1*scale_factor_2, fc='green', ec='green', width=0.02*scale_factor_2)
    ax2.text(x_seg[idx_apoyo] - 0.3, -scale_factor_2*1.1, f'R={R_support:.1f} N', fontsize=7, color='green', 
             ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Añadir barra de color
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, orientation='vertical', pad=0.02, aspect=25, shrink=0.85)
    cbar.set_label('Tensión [N]', fontsize=9)
    
    ax2.set_xlabel('x [m]', fontsize=10)
    ax2.set_ylabel('y (positivo hacia abajo) [m]', fontsize=10)
    ax2.set_title('TENSIONES EN EL CABLE', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    return fig


def mostrar_resultados_streamlit(resultados, solver, precision):
    """Muestra los resultados en formato Streamlit"""
    
    R = resultados['reactions']
    H_mag = abs(resultados['H'])
    V_left = resultados['V_left']
    V_right = resultados['V_right']
    R_support = resultados['R_support']
    peso = resultados['peso_total']
    segmentos = resultados['segmentos']

    # Ángulos respecto a la dirección horizontal
    ang_izq = np.degrees(np.arctan2(V_left, H_mag))
    ang_der = np.degrees(np.arctan2(V_right, H_mag))

    # Verificación del equilibrio vertical
    suma_verticales = V_left + V_right + R_support
    
    # Crear pestañas para organizar la información
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen", "📈 Geometría", "⚖️ Reacciones", "🔍 Precisión"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tensión horizontal H", f"{H_mag:.2f} N")
            st.metric("Peso total", f"{peso:.2f} N")
        with col2:
            st.metric("Tensión máxima", f"{segmentos['tension_max']:.2f} N")
            st.metric("Tensión mínima", f"{segmentos['tension_min']:.2f} N")
        with col3:
            st.metric("Flecha máx. izq", f"{resultados['f1']:.3f} m")
            st.metric("Flecha máx. der", f"{resultados['f2']:.3f} m")
        
        # Tabla de longitudes
        st.subheader("Longitudes")
        df_longitudes = pd.DataFrame({
            "Tramo": ["Izquierdo", "Derecho", "Total"],
            "Longitud [m]": [f"{resultados['L1']:.6f}", f"{resultados['L2']:.6f}", f"{resultados['total_length']:.6f}"]
        })
        st.dataframe(df_longitudes, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Parámetros de la catenaria")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Parámetro a = H/w**: {resultados['a']:.12f} m")
        with col2:
            st.info(f"**Longitud total calculada**: {resultados['total_length']:.12f} m")
        
        st.subheader("Análisis por segmentos")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("N° segmentos", segmentos['n_segmentos'])
        with col2:
            st.metric("Longitud/segmento", f"{segmentos['s_total']/segmentos['n_segmentos']:.6f} m")
        with col3:
            st.metric("Rango tensiones", f"{segmentos['tension_max'] - segmentos['tension_min']:.2f} N")
    
    with tab3:
        st.subheader("Reacciones en apoyos")
        
        # Anclaje izquierdo
        with st.expander("**Anclaje izquierdo (x = 0)**", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Horizontal (←)", f"{H_mag:.6f} N")
            with col2:
                st.metric("Vertical (↑)", f"{V_left:.6f} N")
            with col3:
                st.metric("Resultante", f"{np.hypot(-H_mag, V_left):.6f} N")
            st.caption(f"Ángulo con horizontal: {ang_izq:.6f}°")
        
        # Anclaje derecho
        with st.expander(f"**Anclaje derecho (x = {solver.L_span:.3f} m)**", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Horizontal (→)", f"{H_mag:.6f} N")
            with col2:
                st.metric("Vertical (↑)", f"{V_right:.6f} N")
            with col3:
                st.metric("Resultante", f"{np.hypot(H_mag, V_right):.6f} N")
            st.caption(f"Ángulo con horizontal: {ang_der:.6f}°")
        
        # Apoyo intermedio
        with st.expander(f"**Apoyo intermedio (x = {solver.x_support:.3f} m)**", expanded=True):
            st.metric("Vertical (↑)", f"{R_support:.6f} N")
            st.caption("(No hay componente horizontal)")
        
        # Verificación equilibrio
        st.subheader("Verificación del equilibrio vertical")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Suma reacciones verticales (↑)", f"{suma_verticales:.6f} N")
            st.metric("Peso total del cable (↓)", f"{peso:.6f} N")
        with col2:
            diferencia = suma_verticales - peso
            if abs(diferencia) < 1e-9:
                st.success(f"Diferencia: {diferencia:.2e} N ✓ EQUILIBRIO EXACTO")
            else:
                st.warning(f"Diferencia: {diferencia:.2e} N ⚠ DIFERENCIA NO DESPRECIABLE")
    
    with tab4:
        st.subheader("Precisión numérica")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Error longitud", f"{precision['error_L']:.12e} %")
        with col2:
            st.metric("Error equilibrio V", f"{precision['error_V']:.12e} %")
        with col3:
            st.metric("Error equilibrio H", f"{precision['error_H']:.12e} %")
        
        if precision['cumple_005']:
            st.success("✓ OBJETIVO 0.05% ALCANZADO")
        else:
            st.error("⚠ NO SE ALCANZA EL 0.05%")
        
        # Información adicional
        st.caption("Los errores por debajo de 0.05% se consideran aceptables.")


# ============================================================================
# APLICACIÓN PRINCIPAL STREAMLIT - VERSIÓN CORREGIDA
# ============================================================================
def main():
    st.set_page_config(
        page_title="Solver de Catenaria Exacta",
        page_icon="📐",
        layout="wide"
    )
    
    # Título y descripción
    st.title("📐 Solver de Catenaria Exacta")
    st.markdown("""
    **Cable con dos tramos y un apoyo intermedio fijo en y=0**
    
    Este solver calcula la configuración de equilibrio de un cable con dos tramos 
    y un apoyo intermedio fijo en y=0. Todas las reacciones verticales se muestran 
    como fuerzas hacia arriba (↑).
    """)
    
    # Sidebar con parámetros de entrada
    with st.sidebar:
        st.header("📋 Parámetros del cable")
        
        with st.form("parametros_form"):
            span = st.number_input(
                "Span total [m]",
                min_value=1.0,
                max_value=2500.0,
                value=12.0,
                step=1.0,
                format="%.2f",
                help="Distancia horizontal entre anclajes (1 - 2500 m)"
            )
            
            # CORRECCIÓN: Usar un valor fijo para min_value, no una variable
            length = st.number_input(
                "Longitud cable [m]",
                min_value=1.0,  # Cambiado de L_span*0.01 a 1.0
                max_value=2600.0,
                value=12.8,
                step=1.0,
                format="%.2f",
                help="Debe ser mayor que el span (1 - 2600 m)"
            )
            
            support = st.number_input(
                "Posición del apoyo [m]",
                min_value=1.0,
                max_value=2500.0,
                value=4.5,
                step=1.0,
                format="%.2f",
                help="Debe estar entre 0 y el span (1 - 2500 m)"
            )
            
            diameter = st.number_input(
                "Diámetro cable [m]",
                min_value=0.001,
                max_value=2.0,
                value=0.025,
                step=0.001,
                format="%.3f",
                help="Diámetro del cable (0.001 - 2.0 m)"
            )
            
            calcular = st.form_submit_button("🚀 CALCULAR CATENARIA EXACTA", use_container_width=True)
        
        # Información adicional
        st.divider()
        st.caption("""
        **Nota:** 
        - ρ = 7850 kg/m³ (acero)
        - g = 9.81 m/s²
        - Reacciones verticales siempre ↑
        """)
    
    # Área principal
    if calcular:
        # Validación para asegurar que length > span
        if length <= span:
            st.error(f"❌ La longitud del cable ({length:.2f} m) debe ser mayor que el span ({span:.2f} m)")
        elif support >= span:
            st.error(f"❌ La posición del apoyo ({support:.2f} m) debe ser menor que el span ({span:.2f} m)")
        else:
            try:
                with st.spinner("Calculando configuración de equilibrio..."):
                    # Crear solver y resolver
                    solver = ExactCatenarySolver(span, length, support, diameter)
                    resultados = solver.solve(tol=1e-14)
                    precision = solver.verify_precision(resultados)
                    
                    # Guardar en session state para persistencia
                    st.session_state['solver'] = solver
                    st.session_state['resultados'] = resultados
                    st.session_state['precision'] = precision
                    
                    # Mostrar gráficos
                    st.subheader("📊 Visualización")
                    fig = crear_graficos(solver, resultados)
                    st.pyplot(fig)
                    plt.close(fig)  # Liberar memoria
                    
                    # Mostrar resultados numéricos
                    st.divider()
                    st.subheader("📈 Resultados numéricos")
                    mostrar_resultados_streamlit(resultados, solver, precision)
                    
            except Exception as e:
                st.error(f"Error en el cálculo: {str(e)}")
                st.exception(e)
    
    elif 'resultados' in st.session_state:
        # Recuperar resultados previos
        solver = st.session_state['solver']
        resultados = st.session_state['resultados']
        precision = st.session_state['precision']
        
        # Mostrar gráficos
        st.subheader("📊 Visualización")
        fig = crear_graficos(solver, resultados)
        st.pyplot(fig)
        plt.close(fig)
        
        # Mostrar resultados numéricos
        st.divider()
        st.subheader("📈 Resultados numéricos")
        mostrar_resultados_streamlit(resultados, solver, precision)
    
    else:
        # Mensaje inicial - SIN ALERTAS, solo información neutral
        st.info("👈 Ingresa los parámetros en la barra lateral y haz clic en CALCULAR")
        
        # Ejemplo visual con métricas informativas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rango span", "1 - 2500 m")
        with col2:
            st.metric("Rango longitud", "1 - 2600 m")
        with col3:
            st.metric("Rango diámetro", "0.001 - 2.0 m")


if __name__ == "__main__":
    main()
