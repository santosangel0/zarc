import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dados
try:
    df_clima = pd.read_csv(r'c:\Users\Lenovo\Documents\projects\zarc\dados\analise1_medias_pastagem_goias.csv')
    df_leite = pd.read_csv(r'c:\Users\Lenovo\Documents\projects\zarc\dados\leite_cepea_processado\leite_cepea_goias_mensal.csv')
except Exception as e:
    print(f"Erro ao carregar arquivos: {e}")
    exit()

# Conversão de datas
df_clima['data'] = pd.to_datetime(df_clima['data'])
df_leite['Data_Captacao'] = pd.to_datetime(df_leite['Data_Captacao'])
df_clima = df_clima.sort_values(by='data')
df_leite = df_leite.sort_values(by='Data_Captacao')

# --- 1. DEFLAÇÃO (IPCA) ---
target_col = 'Preco_Medio' # Default
try:
    url_ipca = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=csv'
    df_ipca = pd.read_csv(url_ipca, sep=';')
    df_ipca['data'] = pd.to_datetime(df_ipca['data'], format='%d/%m/%Y')
    df_ipca['valor'] = df_ipca['valor'].str.replace(',', '.').astype(float)
    
    df_ipca = df_ipca[df_ipca['data'] >= '2004-01-01'].copy()
    df_ipca['fator'] = (1 + df_ipca['valor'] / 100)
    df_ipca['indice_acumulado'] = df_ipca['fator'].cumprod()
    indice_base = df_ipca['indice_acumulado'].iloc[-1]
    df_ipca['deflator'] = indice_base / df_ipca['indice_acumulado']
    
    df_leite = pd.merge(df_leite, df_ipca[['data', 'deflator']], left_on='Data_Captacao', right_on='data', how='left')
    df_leite['deflator'] = df_leite['deflator'].ffill() 
    df_leite['Preco_Real'] = df_leite['Preco_Medio'] * df_leite['deflator']
    
    print("Deflação aplicada com sucesso.")
    target_col = 'Preco_Real'
except Exception as e:
    print(f"Erro na deflação: {e}")
    df_leite['Preco_Real'] = df_leite['Preco_Medio']
    target_col = 'Preco_Medio'

# Merge
df_merged = pd.merge(
    df_leite, 
    df_clima, 
    left_on='Data_Captacao', 
    right_on='data', 
    how='inner'
)

# Features
lags_clima = [1, 2, 3, 6]
for col in ['pdsi_medio', 'ndvi_medio', 'itu_medio']:
    for lag in lags_clima:
        df_merged[f'{col}_lag{lag}'] = df_merged[col].shift(lag)

df_merged['Preco_Lag1'] = df_merged[target_col].shift(1)
df_merged['Preco_Lag12'] = df_merged[target_col].shift(12)

df_final = df_merged.dropna().copy()

print(f"Dataset Total: {df_final.shape}")

# --- FILTRAGEM: Pós-Setembro 2022 (Pós-Choque) ---
df_pos_choque = df_final[df_final['Data_Captacao'] >= '2022-09-01'].copy()
print(f"Dataset Pós-Choque (Set/22+): {df_pos_choque.shape}")

if len(df_pos_choque) < 10:
    print("Aviso: Poucos dados para treinar modelo robusto (menos de 10 meses).")
else:
    # Modelagem
    try:
        # Seleção de Features (Garantindo apenas numéricas)
        df_numeric = df_pos_choque.select_dtypes(include=[np.number])
        cols_ignorar = ['ano_captacao', 'mes_captacao', 'ano', 'mes', 'pixels_validos', 'deflator', 'Preco_Medio', 'Preco_Real']
        features = [c for c in df_numeric.columns if c not in cols_ignorar]
        
        print(f"Features utilizadas ({len(features)}): {features}")

        X = df_pos_choque[features].astype(float)
        y = df_pos_choque[target_col].astype(float)

        # Divisão Temporal (80% treino)
        train_size = int(len(df_pos_choque) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        print(f"Treino: {len(X_train)}, Teste: {len(X_test)}")
        
        if len(X_test) > 0:
            # --- MODELO: Random Forest ---
            print("\nTreinando Random Forest (Pós-Choque)...")
            model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
            model_rf.fit(X_train, y_train)
            y_pred_rf = model_rf.predict(X_test)
            mse_rf = mean_squared_error(y_test, y_pred_rf)
            r2_rf = r2_score(y_test, y_pred_rf)
            print(f"RF - R²: {r2_rf:.4f}, MSE: {mse_rf:.4f}")
            
            importances = pd.Series(model_rf.feature_importances_, index=features).sort_values(ascending=False)
            print("\nTop 10 Importância:")
            print(importances.head(10))
        else:
            print("Sem dados suficientes para teste.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Erro na modelagem: {e}")
