"""
Vehicle Classification Model Training
Treina um modelo de machine learning para classificar carros, motos, ônibus e caminhões
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')


class VehicleClassifier:
    """
    Classe para treinar e avaliar um modelo de classificação de veículos
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa o classificador de veículos
        
        Args:
            random_state: Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = ['Carro', 'Moto', 'Ônibus', 'Caminhão']
        
    def load_data(self, filepath):
        """
        Carrega dados do arquivo CSV
        
        Args:
            filepath: Caminho do arquivo CSV com os dados
            
        Returns:
            DataFrame com os dados carregados
        """
        print(f"[INFO] Carregando dados de {filepath}...")
        df = pd.read_csv(filepath)
        print(f"[INFO] Dados carregados: {df.shape[0]} amostras, {df.shape[1]} features")
        return df
    
    def preprocess_data(self, df, target_column='vehicle_type'):
        """
        Realiza pré-processamento dos dados
        
        Args:
            df: DataFrame com os dados
            target_column: Nome da coluna com a classe
            
        Returns:
            X, y processados
        """
        print("[INFO] Iniciando pré-processamento...")
        
        # Separar features e target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Armazenar nomes das features
        self.feature_names = X.columns.tolist()
        
        # Verificar valores faltantes
        print(f"[INFO] Valores faltantes por feature:\n{X.isnull().sum()}")
        X = X.fillna(X.mean(numeric_only=True))
        
        # Remover outliers usando IQR
        print("[INFO] Removendo outliers...")
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
        X = X[mask]
        y = y[mask]
        
        print(f"[INFO] Dados após remoção de outliers: {X.shape[0]} amostras")
        
        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print("[INFO] Classes encontradas:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  - {class_name}: {np.sum(y_encoded == i)} amostras")
        
        return X.reset_index(drop=True), y_encoded
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Divide dados em treino, validação e teste
        
        Args:
            X: Features
            y: Labels
            test_size: Proporção do conjunto de teste
            val_size: Proporção do conjunto de validação
            
        Returns:
            Tupla (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("[INFO] Dividindo dados...")
        
        # Treino e teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Treino e validação
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"[INFO] Treino: {X_train.shape[0]}, Validação: {X_val.shape[0]}, Teste: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_data(self, X_train, X_val, X_test):
        """
        Normaliza os dados usando StandardScaler
        
        Args:
            X_train, X_val, X_test: Conjuntos de features
            
        Returns:
            Tupla (X_train_norm, X_val_norm, X_test_norm)
        """
        print("[INFO] Normalizando dados...")
        
        X_train_norm = self.scaler.fit_transform(X_train)
        X_val_norm = self.scaler.transform(X_val)
        X_test_norm = self.scaler.transform(X_test)
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """
        Treina o modelo
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
            model_type: Tipo de modelo ('random_forest' ou 'gradient_boosting')
            
        Returns:
            Modelo treinado
        """
        print(f"[INFO] Treinando modelo {model_type}...")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Modelo {model_type} não suportado")
        
        self.model.fit(X_train, y_train)
        print("[INFO] Modelo treinado com sucesso!")
        
        return self.model
    
    def evaluate_model(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Avalia o modelo em todos os conjuntos
        
        Args:
            X_train, X_val, X_test: Features
            y_train, y_val, y_test: Labels
        """
        print("\n" + "="*60)
        print("AVALIAÇÃO DO MODELO")
        print("="*60)
        
        # Predições
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)
        
        # Acurácia
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\n[ACURÁCIA]")
        print(f"  Treino: {train_acc:.4f}")
        print(f"  Validação: {val_acc:.4f}")
        print(f"  Teste: {test_acc:.4f}")
        
        # Relatório de classificação
        print(f"\n[RELATÓRIO DE CLASSIFICAÇÃO - TESTE]")
        print(classification_report(
            y_test, y_test_pred, 
            target_names=self.class_names
        ))
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\n[MATRIZ DE CONFUSÃO]")
        print(cm)
        
        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'y_test_pred': y_test_pred,
            'cm': cm
        }
    
    def get_feature_importance(self, top_n=10):
        """
        Retorna as features mais importantes do modelo
        
        Args:
            top_n: Número de features a mostrar
            
        Returns:
            DataFrame com importâncias
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n[TOP {top_n} FEATURES MAIS IMPORTANTES]")
            print(feature_importance_df.head(top_n))
            
            return feature_importance_df
        else:
            print("[AVISO] Modelo não possui feature_importances_")
            return None
    
    def save_model(self, filepath='vehicle_classifier_model.pkl'):
        """
        Salva o modelo treinado
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        print(f"\n[INFO] Salvando modelo em {filepath}...")
        joblib.dump(self.model, filepath)
        print("[INFO] Modelo salvo com sucesso!")
    
    def load_model(self, filepath='vehicle_classifier_model.pkl'):
        """
        Carrega um modelo salvo
        
        Args:
            filepath: Caminho do modelo
        """
        print(f"[INFO] Carregando modelo de {filepath}...")
        self.model = joblib.load(filepath)
        print("[INFO] Modelo carregado com sucesso!")
    
    def predict(self, X):
        """
        Faz predições com o modelo treinado
        
        Args:
            X: Features para predição
            
        Returns:
            Predições (labels codificados)
        """
        X_normalized = self.scaler.transform(X)
        predictions = self.model.predict(X_normalized)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """
        Retorna probabilidades das predições
        
        Args:
            X: Features para predição
            
        Returns:
            Probabilidades para cada classe
        """
        X_normalized = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_normalized)
        return probabilities


def main():
    """
    Função principal para executar o pipeline completo
    """
    print("="*60)
    print("VEHICLE CLASSIFICATION - ML TRAINING PIPELINE")
    print("="*60)
    
    # Inicializar classificador
    classifier = VehicleClassifier(random_state=42)
    
    # Carregar dados (você precisa fornecer um arquivo CSV)
    # Esperado: CSV com coluna 'vehicle_type' e features numéricas
    try:
        df = classifier.load_data('vehicle_data.csv')
    except FileNotFoundError:
        print("[ERRO] Arquivo 'vehicle_data.csv' não encontrado!")
        print("[INFO] Por favor, crie um arquivo CSV com os dados dos veículos")
        return
    
    # Pré-processar dados
    X, y = classifier.preprocess_data(df, target_column='vehicle_type')
    
    # Dividir dados
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.split_data(X, y)
    
    # Normalizar dados
    X_train_norm, X_val_norm, X_test_norm = classifier.normalize_data(
        X_train, X_val, X_test
    )
    
    # Treinar modelo
    classifier.train_model(X_train_norm, y_train, model_type='random_forest')
    
    # Avaliar modelo
    results = classifier.evaluate_model(
        X_train_norm, X_val_norm, X_test_norm,
        y_train, y_val, y_test
    )
    
    # Feature importance
    classifier.get_feature_importance(top_n=15)
    
    # Salvar modelo
    classifier.save_model('vehicle_classifier_model.pkl')
    
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*60)


if __name__ == "__main__":
    main()
