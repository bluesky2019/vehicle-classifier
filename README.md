# 🚗 Vehicle Classifier - Classificação de Veículos com Machine Learning

Um algoritmo completo de Machine Learning para classificar veículos em quatro categorias: **Carros**, **Motos**, **Ônibus** e **Caminhões**.

## 📋 Descrição

Este projeto implementa um pipeline completo de Machine Learning que:
- ✅ Carrega e pré-processa dados de veículos
- ✅ Remove valores faltantes e outliers
- ✅ Normaliza os dados
- ✅ Treina e avalia o modelo
- ✅ Fornece análise de importância de features
- ✅ Realiza predições em novos dados

## 🎯 Características

### Modelos Disponíveis
- **Random Forest** (padrão) - Melhor para este tipo de problema
- **Gradient Boosting** - Alternativa com possível melhor performance

### Pipeline de Treinamento
1. **Carregamento de Dados**: Leitura de arquivo CSV
2. **Pré-processamento**: 
   - Tratamento de valores faltantes
   - Remoção de outliers (IQR)
   - Limpeza de dados
3. **Divisão de Dados**: 70% treino, 10% validação, 20% teste
4. **Normalização**: StandardScaler
5. **Treinamento**: Ajuste do modelo com parâmetros otimizados
6. **Avaliação**: Acurácia, matriz de confusão, relatório de classificação
7. **Salvamento**: Modelo persistido em arquivo .pkl

## 🚀 Como Usar

### 1. Instalação de Dependências

```bash
pip install -r requirements.txt
```

### 2. Preparação de Dados

Você tem duas opções:

#### Opção 1: Usar a Base de Dados Incluída (Recomendado) 🎯
A base de dados `vehicle_data.csv` já está disponibilizada no projeto com **1500 registros** balanceados (375 de cada tipo de veículo). Basta prosseguir para o próximo passo!

#### Opção 2: Criar sua Própria Base de Dados
Crie um arquivo `vehicle_data.csv` com a seguinte estrutura:

```csv
vehicle_type,feature1,feature2,feature3,...
Carro,value1,value2,value3,...
Moto,value1,value2,value3,...
Ônibus,value1,value2,value3,...
Caminhão,value1,value2,value3,...
```

**Campos recomendados:**
- `vehicle_type`: Tipo do veículo (Carro, Moto, Ônibus, Caminhão)
- Features numéricas: peso, comprimento, altura, largura, cilindradas, potência, etc.

### 3. Executar Treinamento

```bash
python train_model.py
```

### 4. Usar o Modelo Treinado

```python
from train_model import VehicleClassifier
import pandas as pd

# Carregar modelo treinado
classifier = VehicleClassifier()
classifier.load_model('vehicle_classifier_model.pkl')

# Fazer predições
novo_dados = pd.DataFrame({
    'feature1': [valor1],
    'feature2': [valor2],
    # ... outras features
})

predicoes = classifier.predict(novo_dados)
print(f"Tipo de veículo: {predicoes[0]}")

# Probabilidades
probs = classifier.predict_proba(novo_dados)
print(f"Confiança: {max(probs[0]):.2%}")
```

## 📊 Estrutura da Classe VehicleClassifier

### Métodos Principais

| Método | Descrição |
|--------|-----------|
| `load_data(filepath)` | Carrega dados de um arquivo CSV |
| `preprocess_data(df)` | Pré-processa e limpa os dados |
| `split_data(X, y)` | Divide em treino, validação e teste |
| `normalize_data(X_train, X_val, X_test)` | Normaliza os dados |
| `train_model(X_train, y_train, model_type)` | Treina o modelo |
| `evaluate_model(...)` | Avalia o desempenho do modelo |
| `get_feature_importance(top_n)` | Mostra features mais importantes |
| `save_model(filepath)` | Salva o modelo treinado |
| `load_model(filepath)` | Carrega um modelo salvo |
| `predict(X)` | Faz predições |
| `predict_proba(X)` | Retorna probabilidades |

## 📈 Saída Esperada

```
============================================================
VEHICLE CLASSIFICATION - ML TRAINING PIPELINE
============================================================
[INFO] Carregando dados de vehicle_data.csv...
[INFO] Dados carregados: 1000 amostras, 10 features
[INFO] Iniciando pré-processamento...
[INFO] Valores faltantes por feature:
feature1    0
feature2    0
...
[INFO] Removendo outliers...
[INFO] Dados após remoção de outliers: 980 amostras
[INFO] Classes encontradas:
  - Carro: 400 amostras
  - Moto: 200 amostras
  - Ônibus: 190 amostras
  - Caminhão: 190 amostras
[INFO] Dividindo dados...
[INFO] Treino: 686, Validação: 98, Teste: 196

============================================================
AVALIAÇÃO DO MODELO
============================================================

[ACURÁCIA]
  Treino: 0.9854
  Validação: 0.9694
  Teste: 0.9592

[RELATÓRIO DE CLASSIFICAÇÃO - TESTE]
              precision    recall  f1-score   support

       Carro       0.96      0.97      0.96        97
        Moto       0.95      0.93      0.94        50
       Ônibus       0.94      0.95      0.94        38
     Caminhão       0.96      0.95      0.95        11

    accuracy                           0.96       196
   macro avg       0.95      0.95      0.95       196
weighted avg       0.96      0.96      0.96       196

[TOP 15 FEATURES MAIS IMPORTANTES]
         feature  importance
       weight       0.2854
       length       0.1963
       height       0.1742
       ...

[INFO] Salvando modelo em vehicle_classifier_model.pkl...
[INFO] Modelo salvo com sucesso!

============================================================
TREINAMENTO CONCLUÍDO COM SUCESSO!
============================================================
```

## 🔧 Parâmetros do Modelo

### Random Forest
```python
n_estimators=100           # Número de árvores
max_depth=15              # Profundidade máxima
min_samples_split=5       # Amostras mínimas para dividir
min_samples_leaf=2        # Amostras mínimas na folha
class_weight='balanced'   # Balanceamento de classes
```

### Gradient Boosting
```python
n_estimators=100          # Número de estimadores
learning_rate=0.1         # Taxa de aprendizado
max_depth=5               # Profundidade máxima
min_samples_split=5       # Amostras mínimas para dividir
```

## 📁 Estrutura do Projeto

```
vehicle-classifier/
├── train_model.py              # Script principal
├── requirements.txt            # Dependências
├── README.md                   # Este arquivo
├── example_usage.py            # Exemplo de uso
├── vehicle_data.csv            # Base de dados (1500 registros)
├── vehicle_classifier_model.pkl # Modelo treinado (gerado)
└── tests/
    └── test_classifier.py      # Testes unitários
```

## 🧪 Testes

Para executar os testes unitários:

```bash
python -m pytest tests/
```

## 💡 Dicas de Otimização

1. **Mais Dados**: Quanto mais dados, melhor a performance
2. **Features Relevantes**: Selecione features que realmente diferenciam os veículos
3. **Balanceamento**: Mantenha um número similar de amostras por classe
4. **Tuning de Hiperparâmetros**: Ajuste os parâmetros conforme necessário
5. **Cross-Validation**: Use validação cruzada para avaliar melhor

## 📝 Formato de Dados Recomendado

```csv
vehicle_type,peso_kg,comprimento_m,altura_m,largura_m,cilindradas_cc,potencia_cv,velocidade_max_kmh,consumo_l_km
Carro,1500,4.5,1.5,1.8,2000,150,200,8.5
Moto,180,2.0,1.0,0.8,600,40,180,25
Ônibus,12000,12,3.5,2.5,9000,300,100,3.5
Caminhão,8000,7,3,2.4,5000,200,120,4.2
```

## 🎓 Conceitos Utilizados

- **Machine Learning**: Aprendizado supervisionado
- **Classificação**: Problema de classificação multiclasse
- **Pré-processamento**: Normalização e tratamento de outliers
- **Validação**: Conjunto de validação e teste separados
- **Avaliação**: Acurácia, matriz de confusão, relatório de classificação
- **Persistência**: Salvamento e carregamento de modelos

## 🚨 Troubleshooting

### Erro: "Arquivo 'vehicle_data.csv' não encontrado"
- Crie o arquivo `vehicle_data.csv` no mesmo diretório que `train_model.py`
- Ou use a base de dados incluída no projeto

### Acurácia baixa
- Verifique a qualidade dos dados
- Aumente o tamanho do conjunto de dados
- Ajuste os hiperparâmetros do modelo

### Desbalanceamento de classes
- O modelo usa `class_weight='balanced'` automaticamente
- Considere fazer oversampling/undersampling

## 📚 Referências

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Feature Scaling Best Practices](https://en.wikipedia.org/wiki/Feature_scaling)

## 📄 Licença

Este projeto é de código aberto e pode ser usado livremente.

## 👤 Autor

Desenvolvido como solução de classificação de veículos usando Machine Learning.

---

**Última atualização**: 2026-05-10

**Status**: ✅ Pronto para uso
