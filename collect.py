import requests
import pandas as pd
from datetime import datetime
import time
import json
import os

os.makedirs('data', exist_ok=True)

print("NHTSA COMPLAINT COLLECTOR")
print("=" * 60)
print("\nColetando reclamações de veículos Ford (2011-2024)\n")

# ============================================
# CONFIGURAÇÃO
# ============================================
BASE_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"

# Modelos Ford para coletar
FORD_MODELS = [
    'FOCUS',      # PowerShift problem
    'FIESTA',     # PowerShift problem
    'F-150',
    'ESCAPE',
    'EXPLORER',
    'EDGE',
    'FUSION',
    'MUSTANG',
    'BRONCO',
    'RANGER'
]

YEARS = range(2011, 2025)

# ============================================
# FUNÇÃO DE COLETA
# ============================================
def collect_complaints(make, model, year):
    """Coleta reclamações para um modelo/ano específico"""
    url = f"{BASE_URL}?make={make}&model={model}&modelYear={year}"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'results' in data and len(data['results']) > 0:
                return data['results']
            else:
                return []
        else:
            print(f"      WARNING: Status {response.status_code}")
            return []
    
    except Exception as e:
        print(f"      ERROR: {e}")
        return []

# ============================================
# COLETA PRINCIPAL
# ============================================
all_complaints = []
total_collected = 0

for model in FORD_MODELS:
    print(f"\n{model}")
    print("-" * 60)
    
    for year in YEARS:
        print(f"   {year}...", end=" ")
        
        complaints = collect_complaints('FORD', model, year)
        
        if complaints:
            # Adiciona metadados
            for complaint in complaints:
                complaint['model_name'] = model
                complaint['model_year'] = year
            
            all_complaints.extend(complaints)
            print(f"OK - {len(complaints)} reclamações")
            total_collected += len(complaints)
        else:
            print("0")
        
        time.sleep(0.5)  # Rate limiting

print("\n" + "=" * 60)
print(f"COLETA CONCLUÍDA: {total_collected:,} reclamações")
print("=" * 60)

# ============================================
# PROCESSA E SALVA
# ============================================
if len(all_complaints) > 0:
    print("\nProcessando dados...")
    
    df = pd.DataFrame(all_complaints)
    
    # Campos importantes
    important_fields = [
        'odiNumber',
        'manufacturer',
        'crash',
        'fire',
        'numberOfInjuries',
        'numberOfDeaths',
        'dateOfIncident',
        'dateComplaintFiled',
        'vin',
        'components',
        'summary',
        'model_name',
        'model_year'
    ]
    
    # Mantém só campos que existem
    available_fields = [f for f in important_fields if f in df.columns]
    df = df[available_fields]
    
    # Converte datas
    if 'dateComplaintFiled' in df.columns:
        df['dateComplaintFiled'] = pd.to_datetime(df['dateComplaintFiled'], errors='coerce')
        df['year_filed'] = df['dateComplaintFiled'].dt.year
        df['month_filed'] = df['dateComplaintFiled'].dt.month
        df['date_filed'] = df['dateComplaintFiled'].dt.date
    
    # Extrai componente principal
    if 'components' in df.columns:
        df['component'] = df['components'].apply(
            lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else 'Unknown'
        )
    
    # Cria texto completo
    df['full_text'] = df['summary'].fillna('')
    
    # Remove duplicatas
    df = df.drop_duplicates(subset=['odiNumber'], keep='first')
    
    # Salva
    output_file = 'data/nhtsa_complaints.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Salvo: {output_file}")
    print(f"Total: {len(df):,} reclamações únicas")
    
    # ============================================
    # ESTATÍSTICAS RÁPIDAS
    # ============================================
    print("\nESTATÍSTICAS:")
    print("Por modelo:")
    for model in df['model_name'].value_counts().head(10).items():
        print(f"   {model[0]}: {model[1]:,}")
    
    print("\nPor ano:")
    if 'year_filed' in df.columns:
        for year in sorted(df['year_filed'].value_counts().items()):
            if pd.notna(year[0]):
                print(f"   {int(year[0])}: {year[1]:,}")
    
    # PowerShift específico
    powershift_models = df[df['model_name'].isin(['FOCUS', 'FIESTA'])].copy()
    powershift_models = powershift_models[
        (powershift_models['model_year'] >= 2011) & 
        (powershift_models['model_year'] <= 2019)
    ]
    
    print(f"\nPowerShift (Focus/Fiesta 2011-2019): {len(powershift_models):,} reclamações")
    
    # Filtra reclamações sobre transmissão
    if 'component' in powershift_models.columns:
        transmission_complaints = powershift_models[
            powershift_models['component'].str.contains('TRANSMISSION', case=False, na=False) |
            powershift_models['full_text'].str.contains('transmission|clutch|shift', case=False, na=False)
        ]
        print(f"   Relacionadas a transmissão: {len(transmission_complaints):,}")
        
        # Salva PowerShift separado
        transmission_complaints.to_csv('data/powershift_complaints.csv', index=False)
        print(f"   Salvo: data/powershift_complaints.csv")
    
    # Salva estatísticas
    stats = {
        'total_complaints': len(df),
        'date_range': {
            'start': str(df['dateComplaintFiled'].min()) if 'dateComplaintFiled' in df.columns else None,
            'end': str(df['dateComplaintFiled'].max()) if 'dateComplaintFiled' in df.columns else None
        },
        'models': df['model_name'].value_counts().to_dict(),
        'powershift_total': len(powershift_models),
        'powershift_transmission': len(transmission_complaints) if 'component' in powershift_models.columns else 0
    }
    
    with open('data/nhtsa_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nEstatísticas salvas: data/nhtsa_stats.json")

else:
    print("\nNenhuma reclamação coletada!")

print("\n" + "=" * 60)
print("PROCESSO CONCLUÍDO")
print("=" * 60)
