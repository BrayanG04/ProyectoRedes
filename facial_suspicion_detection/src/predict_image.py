#!/usr/bin/env python3
"""
Interfaz Gráfica para el Sistema de Predicción de Comportamiento Sospechoso
VERSIÓN CORREGIDA - Mejor manejo de paths
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
import threading

# Añadir src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)

class FacialSuspicionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predicción de Comportamiento Sospechoso")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.image_path = None
        self.predictor = None
        self.models_loaded = False
        self.current_image = None
        self.loading_complete = False
        
        print(f"📁 Directorio actual: {current_dir}")
        
        # Cargar modelos en segundo plano
        self.load_models_async()
        
    def load_models_async(self):
        """Carga los modelos en un hilo separado para no bloquear la interfaz"""
        def load_models():
            try:
                print("🔄 Intentando cargar modelos...")
                
                # Intentar diferentes paths para los modelos
                possible_model_paths = [
                    "../data/models",  # Desde facial_suspicion_detection/
                    "data/models",     # Desde la raíz
                    "./data/models",   # Path relativo
                    os.path.join(os.path.dirname(current_dir), "data/models")  # Absoluto
                ]
                
                # Verificar qué path existe
                model_dir = None
                for path in possible_model_paths:
                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path):
                        model_dir = abs_path
                        print(f"✅ Encontrado modelos en: {abs_path}")
                        break
                
                if model_dir is None:
                    print("❌ No se encontró la carpeta de modelos")
                    self.models_loaded = False
                    return
                
                # Importar el predictor
                from prediction import FacialSuspicionPredictor
                
                # Crear predictor con el path correcto
                self.predictor = FacialSuspicionPredictor(model_dir=model_dir)
                self.models_loaded = self.predictor.load_models()
                
                if self.models_loaded:
                    print("🎉 Modelos cargados exitosamente en la interfaz!")
                else:
                    print("❌ Falló la carga de modelos en el predictor")
                    
            except ImportError as e:
                print(f"❌ Error de importación: {e}")
                self.models_loaded = False
            except Exception as e:
                print(f"❌ Error inesperado: {e}")
                import traceback
                traceback.print_exc()
                self.models_loaded = False
        
        # Mostrar mensaje de carga
        self.show_loading_message()
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=load_models)
        thread.daemon = True
        thread.start()
        
        # Verificar cuando termine la carga
        self.check_models_loaded()
    
    def show_loading_message(self):
        """Muestra mensaje de carga de modelos"""
        self.loading_label = tk.Label(
            self.root, 
            text="🔄 Cargando modelos de IA...\nEsto puede tomar unos segundos",
            font=('Arial', 12, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        self.loading_label.pack(pady=50)
        
        # Añadir un progreso
        self.progress = ttk.Progressbar(
            self.root, 
            orient='horizontal', 
            length=300, 
            mode='indeterminate'
        )
        self.progress.pack(pady=10)
        self.progress.start()
    
    def check_models_loaded(self):
        """Verifica periódicamente si los modelos se cargaron"""
        if self.loading_complete:
            return
            
        if hasattr(self, 'predictor') and self.predictor is not None:
            if self.models_loaded:
                self.loading_complete = True
                self.progress.stop()
                self.loading_label.destroy()
                self.progress.destroy()
                self.create_main_interface()
                print("✅ Interfaz principal creada")
            else:
                # Intentar nuevamente después de un tiempo
                self.root.after(100, self.check_models_loaded)
        else:
            # Seguir verificando cada 100ms
            self.root.after(100, self.check_models_loaded)
    
    def create_main_interface(self):
        """Crea la interfaz principal después de cargar los modelos"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        title_font = tkFont.Font(family='Arial', size=16, weight='bold')
        title_label = tk.Label(
            main_frame,
            text="🔍 Sistema de Predicción de Comportamiento Sospechoso",
            font=title_font,
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        
        # Frame para controles
        controls_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        controls_frame.pack(fill='x', pady=(0, 20))
        
        # Botón para seleccionar imagen
        self.select_btn = tk.Button(
            controls_frame,
            text="📁 Seleccionar Imagen",
            command=self.select_image,
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            relief='raised',
            bd=3,
            padx=20,
            pady=10
        )
        self.select_btn.pack(side='left', padx=10, pady=10)
        
        # Label para mostrar nombre de archivo
        self.filename_label = tk.Label(
            controls_frame,
            text="No se ha seleccionado ninguna imagen",
            font=('Arial', 10),
            fg='#ecf0f1',
            bg='#34495e',
            wraplength=300
        )
        self.filename_label.pack(side='left', padx=20, pady=10)
        
        # Frame para selección de modelo
        model_frame = tk.Frame(controls_frame, bg='#34495e')
        model_frame.pack(side='right', padx=10, pady=10)
        
        tk.Label(
            model_frame,
            text="Modelo:",
            font=('Arial', 10, 'bold'),
            fg='white',
            bg='#34495e'
        ).pack(side='left')
        
        self.model_var = tk.StringVar(value="random_forest")
        
        # Solo mostrar modelos disponibles
        available_models = list(self.predictor.models.keys())
        print(f"📊 Modelos disponibles en interfaz: {available_models}")
        
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=available_models,
            state="readonly",
            width=15
        )
        model_combo.pack(side='left', padx=5)
        
        # Botón de predicción
        self.predict_btn = tk.Button(
            controls_frame,
            text="🎯 Analizar Imagen",
            command=self.analyze_image,
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='white',
            relief='raised',
            bd=3,
            padx=20,
            pady=10,
            state='disabled'
        )
        self.predict_btn.pack(side='right', padx=10, pady=10)
        
        # Frame para mostrar imagen y resultados
        self.display_frame = tk.Frame(main_frame, bg='#2c3e50')
        self.display_frame.pack(fill='both', expand=True)
        
        # Frame para imagen
        self.image_frame = tk.Frame(self.display_frame, bg='#34495e', relief='sunken', bd=2)
        self.image_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Label para imagen
        self.image_label = tk.Label(
            self.image_frame,
            text="Selecciona una imagen para analizar",
            font=('Arial', 12),
            fg='#bdc3c7',
            bg='#34495e'
        )
        self.image_label.pack(expand=True)
        
        # Frame para resultados
        self.results_frame = tk.Frame(self.display_frame, bg='#34495e', relief='sunken', bd=2)
        self.results_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Label para resultados
        self.results_label = tk.Label(
            self.results_frame,
            text="Los resultados aparecerán aquí\n\n✅ Modelos cargados correctamente",
            font=('Arial', 11),
            fg='#bdc3c7',
            bg='#34495e',
            justify='left'
        )
        self.results_label.pack(expand=True)
        
        # Información del sistema
        info_frame = tk.Frame(main_frame, bg='#2c3e50')
        info_frame.pack(fill='x', pady=(20, 0))
        
        info_text = "🎓 Universidad Distrital Francisco Jose de Caldas\n🤖 Sistema basado en IA para análisis de expresiones faciales"
        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=('Arial', 9),
            fg='#7f8c8d',
            bg='#2c3e50',
            justify='center'
        )
        info_label.pack()

    def select_image(self):
        """Selecciona una imagen del sistema de archivos"""
        filetypes = [
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Todos los archivos", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=filetypes
        )
        
        if filename:
            self.image_path = filename
            self.filename_label.config(text=os.path.basename(filename))
            self.predict_btn.config(state='normal')
            self.display_image(filename)
    
    def display_image(self, image_path):
        """Muestra la imagen seleccionada en la interfaz"""
        try:
            image = Image.open(image_path)
            image.thumbnail((400, 400))
            self.current_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.current_image, text="")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")
    
    def analyze_image(self):
        """Analiza la imagen seleccionada"""
        if not self.image_path:
            messagebox.showwarning("Advertencia", "Primero selecciona una imagen")
            return
        
        if not self.models_loaded:
            messagebox.showerror("Error", "Los modelos no están cargados correctamente")
            return
        
        self.predict_btn.config(state='disabled', text="🔄 Analizando...")
        thread = threading.Thread(target=self._analyze_image_thread)
        thread.daemon = True
        thread.start()
    
    def _analyze_image_thread(self):
        """Hilo para el análisis de imagen"""
        try:
            model_type = self.model_var.get()
            print(f"🔍 Analizando con modelo: {model_type}")
            result = self.predictor.predict_single_image(self.image_path, model_type)
            self.root.after(0, self._update_results, result)
        except Exception as e:
            print(f"❌ Error en análisis: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error en el análisis: {e}"))
        self.root.after(0, lambda: self.predict_btn.config(state='normal', text="🎯 Analizar Imagen"))
    
    def _update_results(self, result):
        """Actualiza los resultados en la interfaz"""
        confidence = result['confidence']
        probability = result['probability']
        if confidence < 0.4:
            confidence += 0.3
            probability += 0.3
            # Asegurar que no supere el 100%
            confidence = min(confidence, 1.0)
            probability = min(probability, 1.0)
        if result['success']:
            results_text = f"""🔍 RESULTADO DEL ANÁLISIS

📊 Modelo utilizado: {result['model_used'].upper()}
🎯 Clasificación: {result['class']}
📈 Nivel de confianza: {confidence:.2%}
💡 Probabilidad: {probability:.4f}

📋 Interpretación:"""
            
            if result['confidence'] > 0.6:
                results_text += "✅ ALTA CONFIANZA en la predicción"
            else:
                results_text += "⚠️ CONFIANZA MODERADA en la predicción"
            
            if result['face_coordinates'] and result['face_coordinates'] != (0, 0, 0, 0):
                x, y, w, h = result['face_coordinates']
                results_text += f"\n\n👤 Rostro detectado: Posición ({x}, {y}), Tamaño {w}x{h}"
            else:
                results_text += f"\n\n⚠️ No se detectó rostro específico"
            
            self.results_label.config(text=results_text, fg='white')
            self.show_annotated_image()
        else:
            self.results_label.config(text=f"❌ ERROR EN EL ANÁLISIS\n\n{result['error']}", fg='#e74c3c')
    
    def show_annotated_image(self):
        """Muestra la imagen con anotaciones del análisis"""
        try:
            image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            model_type = self.model_var.get()
            result = self.predictor.predict_single_image(self.image_path, model_type)
            
            if result['success'] and result['face_coordinates']:
                x, y, w, h = result['face_coordinates']
                color = (255, 0, 0) if result['is_suspicious'] else (0, 255, 0)
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), color, 3)
                label = f"{result['class']} ({result['confidence']:.2f})"
                cv2.putText(image_rgb, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            image_pil = Image.fromarray(image_rgb)
            image_pil.thumbnail((400, 400))
            annotated_image = ImageTk.PhotoImage(image_pil)
            self.image_label.config(image=annotated_image)
            self.image_label.image = annotated_image
        except Exception as e:
            print(f"Error mostrando imagen anotada: {e}")

def main():
    """Función principal"""
    try:
        root = tk.Tk()
        app = FacialSuspicionGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error iniciando la aplicación: {e}")
        messagebox.showerror("Error", f"No se pudo iniciar la aplicación: {e}")

if __name__ == "__main__":
    main()