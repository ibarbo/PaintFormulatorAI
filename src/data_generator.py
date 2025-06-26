# PaintFormulatorAI/src/data_generator.py

import pandas as pd
import numpy as np
import os # Importar el módulo os

def generate_paint_data(num_samples=5000):
    """
    Genera un conjunto de datos sintéticos para formulaciones de pintura.

    Args:
        num_samples (int): Número de muestras (formulaciones) a generar.

    Returns:
        pd.DataFrame: DataFrame con los datos sintéticos.
    """

    # --- Definición de ingredientes y rangos ---
    data = {}

    # Componentes principales (porcentajes que suman 100%)
    # Asegurarse de que los rangos son realistas para que no haya NaN en la normalización
    data['resina_pct'] = np.random.uniform(25, 60, num_samples)
    data['pigmento_pct'] = np.random.uniform(5, 35, num_samples)
    data['solvente_pct'] = np.random.uniform(10, 50, num_samples)
    data['aditivo_pct'] = np.random.uniform(0.5, 10, num_samples)

    # Normalizar para que sumen ~100%
    # Sumar los porcentajes actuales y re-escalar
    total_pct = data['resina_pct'] + data['pigmento_pct'] + data['solvente_pct'] + data['aditivo_pct']
    # Evitar división por cero si total_pct es 0 (altamente improbable con uniform)
    total_pct[total_pct == 0] = 1 # Pequeña corrección si por alguna razón la suma fuera 0
    for col in ['resina_pct', 'pigmento_pct', 'solvente_pct', 'aditivo_pct']:
        data[col] = (data[col] / total_pct) * 100 # Escalar para que los porcentajes sumen 100

    # Características de los ingredientes (calidad, tipo)
    data['calidad_resina'] = np.random.choice(['Alta', 'Media', 'Baja'], num_samples, p=[0.4, 0.4, 0.2])
    data['tipo_pigmento'] = np.random.choice(['Dióxido de Titanio', 'Óxido de Hierro', 'Orgánico'], num_samples, p=[0.5, 0.3, 0.2])
    data['tipo_solvente'] = np.random.choice(['Agua', 'Mineral Spirits', 'Acetona', 'Xileno'], num_samples, p=[0.4, 0.3, 0.2, 0.1])
    data['proveedor_aditivo'] = np.random.choice(['ProveedorA', 'ProveedorB', 'ProveedorC', 'ProveedorD'], num_samples, p=[0.3, 0.3, 0.2, 0.2])

    # Condiciones de procesamiento (simuladas)
    data['temperatura_mezcla_C'] = np.random.uniform(20, 60, num_samples)
    data['tiempo_mezcla_min'] = np.random.uniform(30, 180, num_samples)

    # --- Propiedades resultantes (objetivo del formulador) ---
    data['viscosidad_cp'] = np.random.uniform(500, 5000, num_samples)
    data['brillo_unidades'] = np.random.uniform(10, 100, num_samples)
    data['poder_cubriente'] = np.random.randint(1, 11, num_samples)
    data['resistencia_abrasion_ciclos'] = np.random.uniform(100, 1000, num_samples)
    data['estabilidad_almacenamiento_dias'] = np.random.uniform(90, 720, num_samples)

    # --- Creación del DataFrame ---
    df = pd.DataFrame(data)

    # --- Lógica de 'Éxito' o 'Falla' (intencionalmente desbalanceada) ---
    df['exito'] = 0 # 0 para Falla, 1 para Éxito

    # Definir condiciones para el "Éxito"
    condicion_base_exito = (df['resina_pct'] > 35) & (df['resina_pct'] < 55) & \
                           (df['pigmento_pct'] > 15) & (df['pigmento_pct'] < 30) & \
                           (df['solvente_pct'] > 15) & (df['solvente_pct'] < 40) & \
                           (df['aditivo_pct'] > 1) & (df['aditivo_pct'] < 8) & \
                           (df['calidad_resina'] == 'Alta') & \
                           (df['tipo_pigmento'] != 'Orgánico') & \
                           (df['temperatura_mezcla_C'] > 30) & (df['temperatura_mezcla_C'] < 50)


    df.loc[condicion_base_exito, 'exito'] = 1

    # Ajustar algunas propiedades resultantes para que los "Éxitos" tengan mejores valores
    df.loc[df['exito'] == 1, 'viscosidad_cp'] = np.random.uniform(1000, 3000, df['exito'].sum())
    df.loc[df['exito'] == 1, 'brillo_unidades'] = np.random.uniform(70, 95, df['exito'].sum())
    df.loc[df['exito'] == 1, 'poder_cubriente'] = np.random.randint(8, 11, df['exito'].sum())
    df.loc[df['exito'] == 1, 'resistencia_abrasion_ciclos'] = np.random.uniform(600, 1000, df['exito'].sum())
    df.loc[df['exito'] == 1, 'estabilidad_almacenamiento_dias'] = np.random.uniform(365, 720, df['exito'].sum())

    # Introducir variaciones para "Fallas" y ruido para hacer los datos más complejos
    # Algunas combinaciones de baja calidad de resina o tipos de solvente/pigmento pueden llevar a falla
    df.loc[(df['calidad_resina'] == 'Baja') |
           (df['tipo_solvente'] == 'Acetona') |
           (df['pigmento_pct'] < 10) |
           (df['temperatura_mezcla_C'] < 25) | (df['temperatura_mezcla_C'] > 55),
           'exito'] = 0

    # Asegurar el desbalanceo intencionado (ej. forzar más éxitos si quedaron pocos, o viceversa)
    num_exito_actual = df['exito'].sum()
    num_falla_actual = num_samples - num_exito_actual

    target_exito_count = int(num_samples * 0.75) # 75% éxito
    target_falla_count = num_samples - target_exito_count

    if num_exito_actual < target_exito_count:
        # Si tenemos menos éxitos de los deseados, seleccionar algunas fallas y convertirlas en éxito
        idx_to_change = df[df['exito'] == 0].sample(n=(target_exito_count - num_exito_actual), random_state=42).index
        df.loc[idx_to_change, 'exito'] = 1
        # Asegurarse que las propiedades de estas nuevas "exitosas" se ajusten
        df.loc[idx_to_change, 'viscosidad_cp'] = np.random.uniform(1000, 3000, len(idx_to_change))
        df.loc[idx_to_change, 'brillo_unidades'] = np.random.uniform(70, 95, len(idx_to_change))
        df.loc[idx_to_change, 'poder_cubriente'] = np.random.randint(8, 11, len(idx_to_change))
        df.loc[idx_to_change, 'resistencia_abrasion_ciclos'] = np.random.uniform(600, 1000, len(idx_to_change))
        df.loc[idx_to_change, 'estabilidad_almacenamiento_dias'] = np.random.uniform(365, 720, len(idx_to_change))
    elif num_falla_actual < target_falla_count:
        # Si tenemos menos fallas de las deseadas, seleccionar algunos éxitos y convertirlos en falla
        idx_to_change = df[df['exito'] == 1].sample(n=(target_falla_count - num_falla_actual), random_state=42).index
        df.loc[idx_to_change, 'exito'] = 0
        # Asegurarse que las propiedades de estas nuevas "fallas" se ajusten
        df.loc[idx_to_change, 'viscosidad_cp'] = np.random.uniform(500, 5000, len(idx_to_change))
        df.loc[idx_to_change, 'brillo_unidades'] = np.random.uniform(10, 60, len(idx_to_change))
        df.loc[idx_to_change, 'poder_cubriente'] = np.random.randint(1, 7, len(idx_to_change))
        df.loc[idx_to_change, 'resistencia_abrasion_ciclos'] = np.random.uniform(100, 500, len(idx_to_change))
        df.loc[idx_to_change, 'estabilidad_almacenamiento_dias'] = np.random.uniform(90, 360, len(idx_to_change))

    return df

if __name__ == '__main__':
    # Obtener el directorio del script actual
    script_dir = os.path.dirname(__file__)
    # Construir la ruta de salida relativa a la raíz del proyecto
    # Si el script está en src/, y queremos ir a data/raw/ desde la raíz del proyecto,
    # necesitamos retroceder de src/ y luego ir a data/raw/
    project_root = os.path.abspath(os.path.join(script_dir, '..')) # Ir de src/ a PaintFormulatorAI/
    output_dir = os.path.join(project_root, 'data', 'raw') # Luego ir a data/raw/

    # Asegúrate de que la carpeta 'data/raw/' exista
    os.makedirs(output_dir, exist_ok=True)

    synthetic_data = generate_paint_data(num_samples=5000)
    output_path = os.path.join(output_dir, 'simulated_paint_data.csv')
    synthetic_data.to_csv(output_path, index=False)
    print(f"Datos sintéticos generados y guardados en {output_path}")
    print("\nPrimeras 5 filas de los datos:")
    print(synthetic_data.head())
    print(f"\nConteo de Éxito/Falla:\n{synthetic_data['exito'].value_counts()}")
    print(f"\nProporción de Éxito/Falla:\n{synthetic_data['exito'].value_counts(normalize=True)}")