import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
import pandas as pd
import os

class ModelEvaluator:
    def __init__(self, models, feature_extractor):
        self.models = models
        self.feature_extractor = feature_extractor
        
    def evaluate_models(self, X_test, y_test, feature_type='hybrid'):
        """Evalúa todos los modelos - VERSIÓN CORREGIDA"""
        results = {}
        
        # Extraer características si es necesario
        if feature_type != 'raw':
            if feature_type == 'lbp':
                X_test_features = np.array([
                    self.feature_extractor.extract_lbp_features(img) 
                    for img in X_test
                ])
            else:  # hybrid
                X_test_features = self.feature_extractor.extract_hybrid_features(X_test)
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"🔍 EVALUANDO MODELO: {model_name.upper()}")
            print(f"{'='*50}")
            
            try:
                if model_name == 'cnn':
                    # Evaluar CNN
                    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                    
                    # Obtener predicciones
                    y_pred_proba = model.predict(X_test, verbose=0).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # Calcular métricas manualmente
                    test_precision = precision_score(y_test, y_pred, zero_division=0)
                    test_recall = recall_score(y_test, y_pred, zero_division=0)
                    test_f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                else:
                    # Evaluar modelos clásicos
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_features)[:, 1]
                    else:
                        y_pred_proba = model.predict(X_test_features).astype(float)
                    
                    y_pred = model.predict(X_test_features)
                    test_accuracy = accuracy_score(y_test, y_pred)
                    test_precision = precision_score(y_test, y_pred, zero_division=0)
                    test_recall = recall_score(y_test, y_pred, zero_division=0)
                    test_f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Curva ROC
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Métricas adicionales
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Almacenar resultados
                results[model_name] = {
                    'accuracy': test_accuracy,
                    'precision': test_precision,
                    'recall': test_recall,
                    'f1_score': test_f1,
                    'roc_auc': roc_auc,
                    'specificity': specificity,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'confusion_matrix': cm
                }
                
                # Mostrar resultados en consola
                self._print_model_results(model_name, results[model_name])
                
                # Reporte de clasificación
                print("\n📊 REPORTE DE CLASIFICACIÓN:")
                print(classification_report(y_test, y_pred, 
                                          target_names=['No Sospechoso', 'Sospechoso'],
                                          digits=4))
                
                # Visualizaciones
                self.plot_confusion_matrix(cm, model_name)
                self.plot_roc_curve(fpr, tpr, roc_auc, model_name)
                self.plot_precision_recall_curve(y_test, y_pred_proba, model_name)
                
            except Exception as e:
                print(f"❌ Error evaluando {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Comparación final de modelos
        if results:
            self.plot_model_comparison(results)
            self.save_results_to_csv(results)
        
        return results
    
    def _print_model_results(self, model_name, results):
        """Imprime resultados formateados para un modelo"""
        print(f"\n🎯 RESULTADOS - {model_name.upper()}:")
        print(f"   📈 Accuracy:    {results['accuracy']:.4f}")
        print(f"   🎯 Precision:   {results['precision']:.4f}")
        print(f"   🔄 Recall:      {results['recall']:.4f}")
        print(f"   ⚖️  F1-Score:    {results['f1_score']:.4f}")
        print(f"   📊 AUC-ROC:     {results['roc_auc']:.4f}")
        print(f"   🛡️  Specificity: {results['specificity']:.4f}")
    
    def plot_confusion_matrix(self, cm, model_name):
        """Grafica la matriz de confusión"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Sospechoso', 'Sospechoso'],
                   yticklabels=['No Sospechoso', 'Sospechoso'])
        plt.title(f'Matriz de Confusión - {model_name.upper()}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        plt.tight_layout()
        
        # Guardar figura
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, model_name):
        """Grafica la curva ROC"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {model_name.upper()}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Guardar figura
        plt.savefig(f'results/roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_scores, model_name):
        """Grafica la curva Precision-Recall"""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2,
                label=f'Precision-Recall (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva Precision-Recall - {model_name.upper()}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Guardar figura
        plt.savefig(f'results/precision_recall_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results):
        """Compara el rendimiento de todos los modelos"""
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            values = [results[model][metric] for model in models]
            bars = axes[i].bar(models, values, color=colors, alpha=0.8)
            axes[i].set_title(f'Comparación de {metric.upper()}')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=9)
            
            # Mejorar apariencia
            axes[i].set_ylim(0, min(1.0, max(values) * 1.2))
            axes[i].grid(True, alpha=0.3, axis='y')
        
        # Gráfico de comparación general (último subplot)
        if len(axes) > len(metrics):
            axes[len(metrics)].axis('off')
            # Crear tabla de resultados
            metric_data = []
            for model in models:
                row = [model]
                for metric in metrics:
                    row.append(f"{results[model][metric]:.4f}")
                metric_data.append(row)
            
            table = axes[len(metrics)].table(
                cellText=metric_data,
                colLabels=['Modelo'] + [m.upper() for m in metrics],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            axes[len(metrics)].set_title('Tabla Comparativa de Métricas')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_to_csv(self, results):
        """Guarda los resultados en archivo CSV"""
        # Crear DataFrame con todos los resultados
        data = []
        for model_name, model_results in results.items():
            row = {
                'Modelo': model_name,
                'Accuracy': model_results['accuracy'],
                'Precision': model_results['precision'],
                'Recall': model_results['recall'],
                'F1-Score': model_results['f1_score'],
                'AUC-ROC': model_results['roc_auc'],
                'Specificity': model_results['specificity']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        os.makedirs('results', exist_ok=True)
        df.to_csv('results/model_results.csv', index=False, encoding='utf-8')
        
        print(f"\n💾 Resultados guardados en: results/model_results.csv")
        
        # Mostrar tabla resumen en consola
        print("\n" + "="*80)
        print("📋 RESUMEN COMPARATIVO DE MODELOS")
        print("="*80)
        print(df.round(4).to_string(index=False))
        print("="*80)
    
    def bias_analysis(self, X_test, y_test, demographic_data=None):
        """Analiza posibles sesgos demográficos"""
        print("\n" + "="*60)
        print("🔍 ANÁLISIS DE SESGOS Y FAIRNESS")
        print("="*60)
        
        from sklearn.metrics import confusion_matrix
        
        for model_name, model_info in self.models.items():
            if model_name not in self.evaluation_results:
                continue
                
            y_pred = self.evaluation_results[model_name]['predictions']
            
            # Calcular métricas por clase
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Tasa de falsos positivos y negativos
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Precisión por clase
            accuracy_0 = tn / (tn + fp) if (tn + fp) > 0 else 0  # Especificidad
            accuracy_1 = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensibilidad
            
            print(f"\n📊 SESGOS - {model_name.upper()}:")
            print(f"   🔴 Tasa Falsos Positivos: {fpr:.4f}")
            print(f"   🔵 Tasa Falsos Negativos: {fnr:.4f}")
            print(f"   🟢 Precisión Clase 0 (No Sospechoso): {accuracy_0:.4f}")
            print(f"   🟡 Precisión Clase 1 (Sospechoso): {accuracy_1:.4f}")
            print(f"   ⚖️  Diferencia en precisión: {abs(accuracy_0 - accuracy_1):.4f}")
            
            # Análisis de fairness básico
            if abs(accuracy_0 - accuracy_1) > 0.1:
                print("   ⚠️  POSIBLE SESGO: Diferencia significativa entre clases")
            else:
                print("   ✅ BUEN BALANCE: Precisión similar entre clases")
    
    def feature_importance_analysis(self, model, feature_names=None, top_k=15):
        """Analiza la importancia de características (para modelos que lo soportan)"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                if feature_names is None:
                    feature_names = [f'Feature_{i}' for i in range(len(importances))]
                
                # Combinar y ordenar
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Tomar las top_k características
                top_features = feature_importance[:top_k]
                
                # Crear gráfico
                plt.figure(figsize=(12, 8))
                features, importances = zip(*top_features)
                
                y_pos = np.arange(len(features))
                plt.barh(y_pos, importances, align='center', color='skyblue')
                plt.yticks(y_pos, features)
                plt.xlabel('Importancia')
                plt.title(f'Top {top_k} Características Más Importantes')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                # Guardar figura
                plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"\n🎯 Top {top_k} características más importantes:")
                for i, (feature, importance) in enumerate(top_features, 1):
                    print(f"   {i:2d}. {feature}: {importance:.4f}")
                    
        except Exception as e:
            print(f"⚠️ No se pudo analizar importancia de características: {e}")

# Función de utilidad para evaluación rápida
def quick_evaluate(model, X_test, y_test, model_name="Modelo"):
    """Evaluación rápida de un solo modelo"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n🎯 EVALUACIÓN RÁPIDA - {model_name}:")
    print(f"   📈 Accuracy:  {accuracy:.4f}")
    print(f"   🎯 Precision: {precision:.4f}")
    print(f"   🔄 Recall:    {recall:.4f}")
    print(f"   ⚖️  F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }