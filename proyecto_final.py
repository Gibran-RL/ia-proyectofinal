"""
PROYECTO FINAL: CLASIFICACION DE HONGOS (VENENOSOS vs COMESTIBLES)
Facultad de Ciencias, UNAM - Aprendizaje de Maquina

Modelos implementados DESDE CERO siguiendo el material del curso:
  - Arbol de decision (ganancia de informacion / impureza de Gini)
  - Regresion logistica (gradiente descendiente)

sklearn se usa UNICAMENTE para separar los datos (train_test_split) y para
calcular metricas
"""

import os
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')
np.random.seed(12345)

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except Exception:
    pass
sns.set_palette("husl")

if not os.path.exists('outputs'):
    os.makedirs('outputs')

ROJO, AZUL = '#FF6B6B', '#4ECDC4' # <-- para q se vea god 

print("=" * 80)
print("PROYECTO FINAL: CLASIFICACION DE HONGOS (VENENOSOS vs COMESTIBLES)")
print("=" * 80)


# MODELO 1: ARBOL DE DECISION 
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# nota: lo sacamos de las notas del profe
class decisionnode:
    """Nodo del arbol de decision."""
    def __init__(self, feat=-1, value=None, results=None, tb=None, fb=None):
        self.feat = feat        # id de la columna sobre la que se decide
        self.value = value      # valor del rasgo que define la part.
        self.results = results  # counter clases (solo en hojas jeje)
        self.tb = tb            # rama -> cumple el valor"
        self.fb = fb            # rama -> no cumple el valor"

    def __str__(self):
        if self.results is not None:
            return "Class: {}".format(self.results)
        return "Column {}: =={}?".format(self.feat, self.value)


def divideSet(X, column, value):
    """Biparticiona el conjunto segun si la columna toma cierto valor."""
    if isinstance(value, (int, float)):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    set1 = [i for i, row in enumerate(X) if split_function(row)]
    set2 = [j for j, row in enumerate(X) if not split_function(row)]
    return set1, set2


def entropy(classes):
    """Entropia: H(X) = -sum p(y) log2 p(y)."""
    results = Counter(classes)
    H = 0.0
    for r in results.keys():
        p = results[r] / len(classes)
        H -= p * np.log2(p)
    return H


def giniimpurity(classes):
    """Impureza de Gini: G(X) = sum p(y)(1 - p(y))."""
    total = len(classes)
    counts = Counter(classes)
    Gini = 0
    for y, frec in counts.items():
        p = frec / total
        Gini += p * (1 - p)
    return Gini


class DecisionTree():
    """Arbol de decision basado en ganancia de informacion."""
    def __init__(self, score=entropy):
        self.score = score
        self.tree = None

    def buildTree(self, X, Y):
        n, d = X.shape
        current_score = self.score(Y)

        best_gain = 0.0
        best_criteria = None
        best_sets = None
        best_classes = None
        for feature in range(0, d):
            feature_values = set([x[feature] for x in X])
            for value in feature_values:
                set1, set2 = divideSet(X, feature, value)
                p = float(len(set1)) / len(X)
                # Ganancia de informacion: H(X) - E[H(X | rasgo)]
                gain = current_score - p * self.score(Y[set1]) - (1 - p) * self.score(Y[set2])
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (feature, value)
                    best_sets = (X[set1], X[set2])
                    best_classes = (Y[set1], Y[set2])

        if best_gain > 0:
            trueBranch = self.buildTree(best_sets[0], best_classes[0])
            falseBranch = self.buildTree(best_sets[1], best_classes[1])
            return decisionnode(feat=best_criteria[0], value=best_criteria[1],
                                tb=trueBranch, fb=falseBranch)
        else:
            return decisionnode(results=Counter(Y))

    def fit(self, X, Y):
        self.tree = self.buildTree(X, Y)

    def predict(self, observation, subtree=None):
        if self.tree is None:
            raise Exception('Debe entrenarse el arbol. Usar metodo fit(x, y)')
        tree = self.tree if subtree is None else subtree
        if tree.results is not None:
            return list(tree.results.keys())[0]
        v = observation[tree.feat]
        if isinstance(v, (int, float, np.integer, np.floating)):
            branch = tree.tb if v >= tree.value else tree.fb
        else:
            branch = tree.tb if v == tree.value else tree.fb
        return self.predict(observation, branch)


def describe_tree(tree, feature_names, class_names, indent="", lines=None):
    """Construye una lista de lineas legibles que describen el arbol."""
    if lines is None:
        lines = []
    if tree.results is not None:
        clase = list(tree.results.keys())[0]
        lines.append(indent + "--> " + class_names[int(clase)])
    else:
        pregunta = "{} == '{}' ?".format(feature_names[tree.feat], tree.value)
        lines.append(indent + pregunta)
        lines.append(indent + " Si:")
        describe_tree(tree.tb, feature_names, class_names, indent + "   ", lines)
        lines.append(indent + " No:")
        describe_tree(tree.fb, feature_names, class_names, indent + "   ", lines)
    return lines


# MODELO 2: REGRESION LOGISTICA
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# nota: tmb de las notas jaja

# Funcion logistica: f(a) = 1 / (1 + e^-a) toma valores en (0, 1)
logist = lambda a: 1. / (1. + np.exp(-a))


class LogisticRegression():
    """Regresion logistica entrenada con gradiente descendiente."""
    def __init__(self, lr=0.1):
        self.lr = lr            # tasa de aprendizaje (eta)
        self.theta = None       # pesos theta_i
        self.theta0 = 0         # sesgo theta_0

    def fit(self, x, y, max_its=100):
        m, d = x.shape
        self.theta = np.random.rand(d) / np.sqrt(d)
        stop = False
        t = 0
        while not stop:
            for x_i, y_i in zip(x, y):
                f = logist(np.dot(self.theta, x_i) + self.theta0)
                # Regla de actualizacion: theta_i <- theta_i - eta (f(x) - y) x_i
                self.theta = self.theta - self.lr * (f - y_i) * x_i
                self.theta0 = self.theta0 - self.lr * (f - y_i)
            t += 1
            if t > max_its:
                stop = True

    def predict_proba(self, x):
        return logist(np.dot(x, self.theta) + self.theta0)

    def predict(self, x):
        return 1 * (self.predict_proba(x) > 0.5)


# 1. DESCARGA DEL DATASET
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n[1] DESCARGANDO DATASET...")
print("-" * 80)
try:
    mushroom = fetch_ucirepo(id=73)
    X = mushroom.data.features
    # X = mushroom.data.tagets
    y = mushroom.data.targets
    # 'stalk-root' tiene valores faltantes (NaN)... Los tratamos como una categoria
    # mas ('faltante') para q todos los rasgos sean cadenas homogeneas: asi el
    # arbol compara siempre por igualdad y el one-hot genera su propia column
    X = X.fillna('faltante')
    print("Dataset descargado desde UCI Machine Learning Repository (Mushroom, id=73)")
except Exception as e:
    print("Error al descargar el dataset: {}".format(e))
    raise SystemExit

target_col = y.columns[0]
y_series = y[target_col].astype(str)

print("\nProblema identificado: CLASIFICACION BINARIA")
print("Variable objetivo: '{}'".format(target_col))
print("Valores unicos en el objetivo: {}".format(sorted(y_series.unique())))


# 2. DESCRIPCION DEL DATASET
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n[2] DESCRIPCION DEL DATASET")
print("-" * 80)

n_samples, n_features = X.shape
print("\nNumero total de muestras: {}".format(n_samples))
print("Numero de rasgos (features): {}".format(n_features))
print("Rasgos: {}".format(list(X.columns)))

print("\nDistribucion de la variable objetivo:")
print(y_series.value_counts().to_string())
print("\nProporcion:")
print((y_series.value_counts(normalize=True) * 100).round(2).to_string())

# Descripcion estadistica de rasgos categoricos (no aplica media/rango numerico)
print("\nDescripcion estadistica (rasgos categoricos):")
print(X.describe().T[['count', 'unique', 'top', 'freq']].to_string())

print("\nFrecuencia de los primeros 3 rasgos:")
for col in X.columns[:3]:
    print("\n{}:".format(col))
    print(X[col].value_counts().to_string())

# Codificacion de la clase objetivo a 0/1 (venenoso = 1, la clase de riesgo)
clase_positiva = 'p' if 'p' in set(y_series.unique()) else sorted(y_series.unique())[-1]
y01 = (y_series == clase_positiva).astype(int).values
NOMBRES_CLASE = {0: 'Comestible', 1: 'Venenoso'}
print("\nCodificacion del objetivo: '{}'=1 (Venenoso), resto=0 (Comestible)".format(clase_positiva))

# Particion 70% / 15% / 15% (estratificada) 
idx = np.arange(n_samples)
idx_temp, idx_test = train_test_split(idx, test_size=0.15, random_state=42, stratify=y01)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.15 / 0.85,
                                      random_state=42, stratify=y01[idx_temp])

print("\nDivision del dataset:")
print("  - Entrenamiento: {} muestras (~70%)".format(len(idx_train)))
print("  - Validacion:    {} muestras (~15%)".format(len(idx_val)))
print("  - Prueba (test): {} muestras (~15%)".format(len(idx_test)))

#   Visual 1: distribucion de clases 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
y_counts = y_series.value_counts()
etiquetas = [NOMBRES_CLASE[int(v == clase_positiva)] for v in y_counts.index]
axes[0].bar(etiquetas, y_counts.values, color=[AZUL, ROJO], alpha=0.8, edgecolor='black')
axes[0].set_title('Distribucion de Clases (absoluta)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Cantidad de muestras')
for i, v in enumerate(y_counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')
axes[1].pie(y_counts.values, labels=etiquetas, autopct='%1.1f%%',
            colors=[AZUL, ROJO], startangle=90, textprops={'fontweight': 'bold'})
axes[1].set_title('Distribucion de Clases (porcentaje)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/01_distribucion_clases.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGrafica guardada: outputs/01_distribucion_clases.png")

#  Visual 2: frecuencia de rasgos 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
for i, col in enumerate(X.columns[:4]):
    counts = X[col].value_counts()
    axes[i].barh(range(len(counts)), counts.values, color='steelblue',
                 alpha=0.8, edgecolor='black')
    axes[i].set_yticks(range(len(counts)))
    axes[i].set_yticklabels(counts.index)
    axes[i].set_title("Frecuencia: {}".format(col), fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Cantidad')
plt.tight_layout()
plt.savefig('outputs/02_frecuencia_rasgos.png', dpi=150, bbox_inches='tight')
plt.close()
print("Grafica guardada: outputs/02_frecuencia_rasgos.png")


# 3. PREPARACION DE DATOS
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n[3] PREPARACION DE DATOS")
print("-" * 80)
print("""
Los dos modelos necesitan representaciones distintas:

  - ARBOL DE DECISION: trabaja directamente con los rasgos CATEGORICOS
    (cadenas de texto). El arbol parte el conjunto preguntando '¿rasgo == valor?',
    asi que NO se codifica nada.

  - REGRESION LOGISTICA: es un modelo lineal y calcula theta . x. Los rasgos son
    categoricos NOMINALES (no tienen orden), por lo que se aplica codificacion
    ONE-HOT (una columna binaria por categoria). Asignar enteros 0,1,2,... a las
    categorias introduciria un orden falso que el modelo lineal tomaria literal.
""")

# para el TREE arreglos numpy de cadenas (datos crudos)
X_np = X.values
Xtr_tree, Xva_tree, Xte_tree = X_np[idx_train], X_np[idx_val], X_np[idx_test]

# para la regresion logistica: one-hot... colum tomadas del entrenamiento
X_train_df = X.iloc[idx_train]
cols_oh = pd.get_dummies(X_train_df).columns
def a_one_hot(df):
    return pd.get_dummies(df).reindex(columns=cols_oh, fill_value=0).astype(float).values
Xtr_oh = a_one_hot(X_train_df)
Xva_oh = a_one_hot(X.iloc[idx_val])
Xte_oh = a_one_hot(X.iloc[idx_test])

ytr, yva, yte = y01[idx_train], y01[idx_val], y01[idx_test]
print("Arbol: {} rasgos categoricos | Regresion logistica: {} columnas one-hot"
      .format(Xtr_tree.shape[1], Xtr_oh.shape[1]))


#    JUSTIFICACION DE LOS MODELOS
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n[3.1] JUSTIFICACION DE LOS MODELOS")
print("-" * 80)
print("""
Se eligen dos modelos vistos en el curso que abordan la clasificacion desde
paradigmas distintos, lo que hace la comparacion mas informativa:

1. ARBOL DE DECISION (modelo NO parametrico):
   - Clasifica con una jerarquia de reglas '¿rasgo == valor?', muy natural para
     datos categoricos como los de los hongos.
   - Es directamente interpretable: el arbol resultante son las reglas mismas.
   - No asume forma de la frontera de decision ni necesita codificacion.

2. REGRESION LOGISTICA (modelo parametrico / lineal):
   - Estima p(venenoso | x) con la funcion logistica sobre una combinacion
     lineal de los rasgos.
   - Aprende un peso theta_i por categoria; esos pesos son el "estado de
     creencias" del modelo: que tanto empuja cada categoria hacia 'venenoso'.
   - Sirve de contraste: un modelo lineal frente a uno basado en reglas.
""")


# 4. ENTRENAMIENTO Y AJUSTE DE HIPERPARAMETROS (en validacion)
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n[4] ENTRENAMIENTO CON AJUSTE DE HIPERPARAMETROS")
print("-" * 80)

#  ARBOL DE DECISION: criterio de particion (entropia vs Gini) 
print("\n[4.1] ARBOL DE DECISION")
print("-" * 40)
print("Hiperparametro a ajustar: criterio de particion (ganancia de informacion")
print("con ENTROPIA vs IMPUREZA DE GINI). Se elige el mejor en validacion.\n")

candidatos_arbol = {'entropy': entropy, 'gini': giniimpurity}
mejor_crit, mejor_acc_arbol, mejor_arbol = None, -1, None
for nombre, fn in candidatos_arbol.items():
    t0 = time.time()
    arbol = DecisionTree(score=fn)
    arbol.fit(Xtr_tree, ytr)
    pred_val = np.array([arbol.predict(x) for x in Xva_tree])
    acc = accuracy_score(yva, pred_val)
    print("  criterio={:8s} -> exactitud validacion = {:.4f}  ({:.1f}s)"
          .format(nombre, acc, time.time() - t0))
    if acc > mejor_acc_arbol:
        mejor_acc_arbol, mejor_crit, mejor_arbol = acc, nombre, arbol

dt_model = mejor_arbol
print("\nCriterio seleccionado: {} (exactitud validacion = {:.4f})"
      .format(mejor_crit, mejor_acc_arbol))

#  REGRESION LOGISTICA: tasa de aprendizaje e iteraciones 
print("\n[4.2] REGRESION LOGISTICA")
print("-" * 40)
print("Hiperparametros a ajustar: tasa de aprendizaje (lr) y numero de")
print("iteraciones (max_its). Se elige la mejor combinacion en validacion.\n")

grid_lr = [0.01, 0.1, 0.5]
grid_its = [50, 100]
mejor_params_lr, mejor_acc_lr, mejor_lr_model = None, -1, None
for lr in grid_lr:
    for its in grid_its:
        np.random.seed(12345)
        t0 = time.time()
        modelo = LogisticRegression(lr=lr)
        modelo.fit(Xtr_oh, ytr, max_its=its)
        acc = accuracy_score(yva, modelo.predict(Xva_oh))
        print("  lr={:<5} max_its={:<4} -> exactitud validacion = {:.4f}  ({:.1f}s)"
              .format(lr, its, acc, time.time() - t0))
        if acc > mejor_acc_lr:
            mejor_acc_lr, mejor_params_lr, mejor_lr_model = acc, (lr, its), modelo

lr_model = mejor_lr_model
print("\nHiperparametros seleccionados: lr={}, max_its={} (exactitud validacion = {:.4f})"
      .format(mejor_params_lr[0], mejor_params_lr[1], mejor_acc_lr))


# 5. EVALUACION EN EL CONJUNTO DE PRUEBA
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n[5] EVALUACION Y COMPARACION (conjunto de prueba)")
print("-" * 80)

y_pred_dt = np.array([dt_model.predict(x) for x in Xte_tree])
y_pred_lr = lr_model.predict(Xte_oh)


def metricas(y_true, y_pred):
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }


m_dt = metricas(yte, y_pred_dt)
m_lr = metricas(yte, y_pred_lr)


def imprime_metricas(nombre, m):
    print("\n{:^60}".format(nombre))
    print("-" * 60)
    print("Exactitud (Accuracy):  {:.4f}  ({:.2f}%)".format(m['accuracy'], m['accuracy'] * 100))
    print("Precision (venenoso):  {:.4f}".format(m['precision']))
    print("Recall    (venenoso):  {:.4f}".format(m['recall']))
    print("F1-Score:              {:.4f}".format(m['f1']))


print("\nMETRICAS EN PRUEBA (clase positiva = Venenoso):")
print("=" * 60)
imprime_metricas("ARBOL DE DECISION", m_dt)
imprime_metricas("REGRESION LOGISTICA", m_lr)

cm_dt = confusion_matrix(yte, y_pred_dt)
cm_lr = confusion_matrix(yte, y_pred_lr)
print("\nMATRICES DE CONFUSION (filas=real, columnas=prediccion):")
print("\nArbol de decision:\n", cm_dt)
print("\nRegresion logistica:\n", cm_lr)

#  Grafica: comparacion de metricas 
nombres_m = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
val_dt = [m_dt['accuracy'], m_dt['precision'], m_dt['recall'], m_dt['f1']]
val_lr = [m_lr['accuracy'], m_lr['precision'], m_lr['recall'], m_lr['f1']]
x = np.arange(len(nombres_m)); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w / 2, val_dt, w, label='Arbol de decision', color=ROJO, alpha=0.85, edgecolor='black')
ax.bar(x + w / 2, val_lr, w, label='Regresion logistica', color=AZUL, alpha=0.85, edgecolor='black')
ax.set_ylabel('Puntuacion', fontweight='bold')
ax.set_title('Comparacion de metricas en prueba', fontsize=12, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(nombres_m)
ax.set_ylim([0, 1.05]); ax.legend(); ax.grid(axis='y', alpha=0.3)
for i, (a, b) in enumerate(zip(val_dt, val_lr)):
    ax.text(i - w / 2, a + 0.01, "{:.3f}".format(a), ha='center', fontsize=8)
    ax.text(i + w / 2, b + 0.01, "{:.3f}".format(b), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('outputs/03_comparacion_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGrafica guardada: outputs/03_comparacion_metricas.png")

#  Grafica: matrices de confusion 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
etq = ['Comestible', 'Venenoso']
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Reds', ax=axes[0],
            xticklabels=etq, yticklabels=etq, cbar=False)
axes[0].set_title('Matriz de confusion - Arbol de decision', fontweight='bold')
axes[0].set_ylabel('Real'); axes[0].set_xlabel('Prediccion')
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=etq, yticklabels=etq, cbar=False)
axes[1].set_title('Matriz de confusion - Regresion logistica', fontweight='bold')
axes[1].set_ylabel('Real'); axes[1].set_xlabel('Prediccion')
plt.tight_layout()
plt.savefig('outputs/04_matrices_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("Grafica guardada: outputs/04_matrices_confusion.png")


# 5.1 INTERPRETABILIDAD DE LOS MODELOS
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n[5.1] INTERPRETABILIDAD")
print("-" * 80)

# tree: reglas aprendidas (se guardan completas, se muestra el inicio)
lineas_arbol = describe_tree(dt_model.tree, list(X.columns), NOMBRES_CLASE)
rasgo_raiz = X.columns[dt_model.tree.feat] if dt_model.tree.results is None else None
print("\nReglas del arbol (primer rasgo de decision: '{}'):".format(rasgo_raiz))
print("\n".join(lineas_arbol[:12]))
if len(lineas_arbol) > 12:
    print("   ... (arbol completo en outputs/arbol_decision.txt)")
with open('outputs/arbol_decision.txt', 'w', encoding='utf-8') as f:
    f.write("ARBOL DE DECISION (criterio: {})\n".format(mejor_crit))
    f.write("=" * 60 + "\n")
    f.write("\n".join(lineas_arbol))

# rgresion logistica: pesos theta = "estado de creencias"
pesos = pd.Series(lr_model.theta, index=cols_oh).sort_values()
print("\nRegresion logistica - rasgos que mas empujan hacia VENENOSO (theta > 0):")
for nombre, val in pesos.tail(5)[::-1].items():
    print("  {:30s}  theta = {:+.3f}".format(nombre, val))
print("\nRasgos que mas empujan hacia COMESTIBLE (theta < 0):")
for nombre, val in pesos.head(5).items():
    print("  {:30s}  theta = {:+.3f}".format(nombre, val))

# grafica de pesos mas influyentes (|theta|)
top = pesos.reindex(pesos.abs().sort_values().tail(15).index)
fig, ax = plt.subplots(figsize=(10, 7))
colores = [ROJO if v > 0 else AZUL for v in top.values]
ax.barh(range(len(top)), top.values, color=colores, alpha=0.85, edgecolor='black')
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Peso theta  (>0 empuja a Venenoso, <0 a Comestible)', fontweight='bold')
ax.set_title('Regresion logistica: rasgos mas influyentes', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/05_pesos_regresion.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGrafica guardada: outputs/05_pesos_regresion.png")


# 6. CONCLUSIONES Y ANALISIS CUALITATIVO
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n[6] CONCLUSIONES")
print("=" * 80)

if abs(m_dt['f1'] - m_lr['f1']) < 1e-4:
    veredicto = "Ambos modelos obtienen practicamente el mismo desempeno."
elif m_dt['f1'] > m_lr['f1']:
    veredicto = "El arbol de decision obtuvo un F1 ligeramente mayor."
else:
    veredicto = "La regresion logistica obtuvo un F1 ligeramente mayor."

top_venenoso = ", ".join(pesos.tail(3)[::-1].index)

conclusiones = """
ANALISIS CUALITATIVO

1. DESEMPENO GENERAL
   {linea}
   Arbol de decision   -> Accuracy {acc_dt:.4f} | F1 {f1_dt:.4f}
   Regresion logistica -> Accuracy {acc_lr:.4f} | F1 {f1_lr:.4f}

   {veredicto} Esto era esperable: el conjunto de hongos es casi perfectamente
   separable a partir de los rasgos categoricos, por lo que dos modelos de
   naturaleza muy distinta llegan ambos a una exactitud muy alta. La conclusion
   relevante NO es "cual gana por unas decimas", sino que el problema es
   linealmente/estructuralmente facil y que el verdadero contraste esta en la
   INTERPRETABILIDAD.

2. QUE APRENDIO CADA MODELO
   {linea}
   - El arbol de decision toma como primera regla el rasgo '{raiz}', lo que
     indica que por si solo separa gran parte de las clases. El arbol completo
     (outputs/arbol_decision.txt) es una lista de reglas '¿rasgo == valor?'
     directamente legible.
   - La regresion logistica coincide en lo esencial: las categorias con mayor
     peso positivo hacia 'venenoso' son {top_ven}. Sus pesos theta dan una
     lectura cuantitativa de cuanto contribuye cada categoria.

3. INTERPRETACION DE LAS METRICAS (clase positiva = Venenoso)
   {linea}
   En este dominio el RECALL de la clase 'venenoso' es la metrica critica: un
   falso negativo (clasificar como comestible un hongo venenoso) es el error
   peligroso. Conviene leer la matriz de confusion priorizando esa celda por
   encima de la exactitud global.

4. LIMITACIONES
   {linea}
   - El dataset solo tiene rasgos categoricos; no hay medidas continuas (tamano,
     peso) que podrian ser necesarias en casos mas dificiles.
   - La alta separabilidad hace que las metricas sean optimistas; en un problema
     mas ruidoso la diferencia entre modelos seria mayor.
   - El arbol implementado crece hasta separar por completo (sin poda ni
     profundidad maxima), por lo que con datos ruidosos tenderia al sobreajuste.
   - La regresion logistica usa gradiente descendiente basico; sobre datos
     separables los pesos crecen sin cota teorica (aqui se controla con un numero
     fijo de iteraciones).
   - Se uso una particion fija con validacion estratificada; una validacion
     cruzada k-fold daria una estimacion mas robusta.

5. RECOMENDACIONES
   {linea}
   - Para clasificar nuevos hongos basta cualquiera de los dos modelos; se
     preferiria el arbol si se necesita explicar la decision con reglas, y la
     regresion logistica si se quiere una probabilidad calibrada.
   - Ante un costo asimetrico (es peor liberar un venenoso), subir el umbral de
     decision de la regresion logistica para favorecer el recall de 'venenoso'.
   - Validar con k-fold y, de ser posible, con datos de otras regiones.
""".format(
    linea="-" * 74,
    acc_dt=m_dt['accuracy'], f1_dt=m_dt['f1'],
    acc_lr=m_lr['accuracy'], f1_lr=m_lr['f1'],
    veredicto=veredicto, raiz=rasgo_raiz, top_ven=top_venenoso,
)

print(conclusiones)

with open('outputs/CONCLUSIONES.txt', 'w', encoding='utf-8') as f:
    f.write("PROYECTO FINAL: CLASIFICACION DE HONGOS\n")
    f.write("=" * 80 + "\n")
    f.write(conclusiones)

print("=" * 80)
print("PROYECTO COMPLETADO")
print("=" * 80)
print("\nArchivos generados en outputs/:")
for nombre in ["01_distribucion_clases.png", "02_frecuencia_rasgos.png",
               "03_comparacion_metricas.png", "04_matrices_confusion.png",
               "05_pesos_regresion.png", "arbol_decision.txt", "CONCLUSIONES.txt"]:
    print("  - outputs/{}".format(nombre))
