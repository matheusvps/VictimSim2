import matplotlib.pyplot as plt
import numpy as np

# ------------------- DADOS GERAIS -------------------
# Vítimas no ambiente
labels = ['Críticas', 'Instáveis', 'Pot. Instáveis', 'Estáveis']
total = {'Críticas': 78, 'Instáveis': 159, 'Pot. Instáveis': 38, 'Estáveis': 25}
total_vitimas = sum(total.values())

# Encontradas por exploradores (acumulado)
encontradas = {'Críticas': 28, 'Instáveis': 48, 'Pot. Instáveis': 15, 'Estáveis': 10}
total_encontradas = sum(encontradas.values())

# Salvas por resgatadores (acumulado)
salvas = {'Críticas': 17, 'Instáveis': 36, 'Pot. Instáveis': 9, 'Estáveis': 8}
total_salvas = sum(salvas.values())

# Percentuais gerais
percent_encontradas = {k: 100*v/total[k] for k, v in encontradas.items()}
percent_salvas = {k: 100*v/total[k] for k, v in salvas.items()}

# Gravidade total
gravidade_total = 11252.89
gravidade_encontradas = 3933.00
gravidade_salvas = 2808.95

# ------------------- DADOS POR AGENTE -------------------
# Exploradores
exploradores = {
    'EXPL_1': {'Críticas': 6, 'Instáveis': 15, 'Pot. Instáveis': 7, 'Estáveis': 4, 'Veg': 0.0946},
    'EXPL_2': {'Críticas': 8, 'Instáveis': 15, 'Pot. Instáveis': 4, 'Estáveis': 2, 'Veg': 0.0985},
    'EXPL_3': {'Críticas': 9, 'Instáveis': 12, 'Pot. Instáveis': 6, 'Estáveis': 1, 'Veg': 0.0985},
    'EXPL_4': {'Críticas': 15, 'Instáveis': 16, 'Pot. Instáveis': 7, 'Estáveis': 4, 'Veg': 0.1491},
}

# Resgatadores
rescatadores = {
    'RESC_1': {'Críticas': 1, 'Instáveis': 1, 'Pot. Instáveis': 1, 'Estáveis': 0, 'Vsg': 0.0105},
    'RESC_2': {'Críticas': 3, 'Instáveis': 18, 'Pot. Instáveis': 2, 'Estáveis': 5, 'Vsg': 0.0774},
    'RESC_3': {'Críticas': 8, 'Instáveis': 11, 'Pot. Instáveis': 5, 'Estáveis': 0, 'Vsg': 0.0870},
    'RESC_4': {'Críticas': 6, 'Instáveis': 7, 'Pot. Instáveis': 2, 'Estáveis': 3, 'Vsg': 0.0612},
}

# ------------------- PLOT EXPLORADORES -------------------
fig, ax = plt.subplots(figsize=(10,6))
bar_width = 0.18
x = np.arange(len(labels))

for i, (agente, dados) in enumerate(exploradores.items()):
    valores = [dados[k] for k in labels]
    ax.bar(x + i*bar_width - 1.5*bar_width, valores, bar_width, label=agente)

ax.set_ylabel('Vítimas encontradas')
ax.set_title('Vítimas encontradas por cada explorador')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()

# Percentual de vítimas encontradas por explorador
fig, ax = plt.subplots(figsize=(10,6))
for i, (agente, dados) in enumerate(exploradores.items()):
    percentuais = [100*dados[k]/total[k] for k in labels]
    ax.bar(x + i*bar_width - 1.5*bar_width, percentuais, bar_width, label=agente)

ax.set_ylabel('Percentual (%)')
ax.set_title('Percentual de vítimas encontradas por explorador')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()

# ------------------- PLOT RESGATADORES -------------------
fig, ax = plt.subplots(figsize=(10,6))
for i, (agente, dados) in enumerate(rescatadores.items()):
    valores = [dados[k] for k in labels]
    ax.bar(x + i*bar_width - 1.5*bar_width, valores, bar_width, label=agente)

ax.set_ylabel('Vítimas salvas')
ax.set_title('Vítimas salvas por cada resgatador')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()

# Percentual de vítimas salvas por resgatador
fig, ax = plt.subplots(figsize=(10,6))
for i, (agente, dados) in enumerate(rescatadores.items()):
    percentuais = [100*dados[k]/total[k] for k in labels]
    ax.bar(x + i*bar_width - 1.5*bar_width, percentuais, bar_width, label=agente)

ax.set_ylabel('Percentual (%)')
ax.set_title('Percentual de vítimas salvas por resgatador')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()

# ------------------- PLOT GERAIS (ACUMULADOS) -------------------
# 1. Gráfico de barras: Número de vítimas por severidade
fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - 0.25, [total[k] for k in labels], 0.25, label='No ambiente')
rects2 = ax.bar(x, [encontradas[k] for k in labels], 0.25, label='Encontradas')
rects3 = ax.bar(x + 0.25, [salvas[k] for k in labels], 0.25, label='Salvas')

ax.set_ylabel('Número de vítimas')
ax.set_title('Vítimas por severidade (acumulado)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()

# 2. Gráfico de barras: Percentual de vítimas encontradas e salvas
fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - 0.125, [percent_encontradas[k] for k in labels], 0.25, label='Encontradas (%)')
rects2 = ax.bar(x + 0.125, [percent_salvas[k] for k in labels], 0.25, label='Salvas (%)')

ax.set_ylabel('Percentual (%)')
ax.set_title('Percentual de vítimas encontradas e salvas por severidade (acumulado)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.show()

# 3. Gravidade total
fig, ax = plt.subplots(figsize=(7,5))
gravidades = [gravidade_total, gravidade_encontradas, gravidade_salvas]
grav_labels = ['Total no ambiente', 'Encontradas', 'Salvas']
ax.bar(grav_labels, gravidades, color=['gray', 'orange', 'green'])
ax.set_ylabel('Soma das gravidades')
ax.set_title('Gravidade total das vítimas')
plt.tight_layout()
plt.show()