
import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# Configuração de Caminhos
# ---------------------------------------------------------
input_file = '../dados/leite_cepea_bruto/leite_cepea_bruto.csv'
output_dir = '../dados/leite_cepea_processado'
output_file = os.path.join(output_dir, 'leite_cepea_goias_mensal.csv')

os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# Leitura e Filtragem
# ---------------------------------------------------------
# O arquivo bruto possui cabeçalho na linha 4 (índice 3)
try:
    df = pd.read_csv(input_file, header=3)
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em {input_file}")
    raise

# Dicionário de meses
mes_map = {
    'JAN': 1, 'FEV': 2, 'MAR': 3, 'ABR': 4, 'MAI': 5, 'JUN': 6,
    'JUL': 7, 'AGO': 8, 'SET': 9, 'OUT': 10, 'NOV': 11, 'DEZ': 12
}

# Filtrar Goiás
df_go = df[df['Estado'] == 'GO'].copy()

# Criar Data de Referência (Original)
def parse_mes(m):
    return mes_map.get(str(m).upper(), 0)

df_go['Mes_Num'] = df_go['Mes'].apply(parse_mes)
df_go['Data_Referencia'] = pd.to_datetime(
    df_go[['Ano', 'Mes_Num']].assign(day=1).rename(columns={'Ano': 'year', 'Mes_Num': 'month'})
)

# ---------------------------------------------------------
# Aplicação das Regras de Negócio
# ---------------------------------------------------------
def ajustar_data_captacao(row):
    data_ref = row['Data_Referencia']
    
    # Transição: Até 2022 (inclusive) deduzir 1 mês. A partir de 2023 manter.
    if data_ref.year < 2023:
        return data_ref - pd.DateOffset(months=1)
    else:
        return data_ref

df_go['Data_Captacao'] = df_go.apply(ajustar_data_captacao, axis=1)

# Seleção e Ordenação final
df_final = df_go[['Data_Captacao', 'Preco_Medio']].copy()
df_final['ano_captacao'] = df_final['Data_Captacao'].dt.year
df_final['mes_captacao'] = df_final['Data_Captacao'].dt.month
df_final = df_final.sort_values(by='Data_Captacao').reset_index(drop=True)

# ---------------------------------------------------------
# Validação e Exportação
# ---------------------------------------------------------
print(f"Série Processada para Goiás (Total: {len(df_final)} registros)")
if not df_final.empty:
    print(f"Início da Série: {df_final['Data_Captacao'].min().strftime('%m/%Y')}")
    print(f"Fim da Série:    {df_final['Data_Captacao'].max().strftime('%m/%Y')}")

    # Verificar lacunas
    range_full = pd.date_range(
        start=df_final['Data_Captacao'].min(), 
        end=df_final['Data_Captacao'].max(), 
        freq='MS'
    )
    datas_presentes = set(df_final['Data_Captacao'])
    lacunas = [d for d in range_full if d not in datas_presentes]

    if lacunas:
        print("\n[ALERTA] Lacunas identificadas na série temporal (Meses faltantes):")
        for d in lacunas:
            print(f" - {d.strftime('%m/%Y')}")
        print("Nota: A lacuna em 12/2022 é esperada devido à transição metodológica.")

# Salvar
df_final.to_csv(output_file, index=False)
print(f"\nArquivo exportado com sucesso para: {output_file}")
