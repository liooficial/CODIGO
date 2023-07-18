import csv
import pickle
import json
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import sys

#corpus = ['iclr', 'acl', 'arxiv_cs_ai', 'arxiv_cs_cl', 'arxiv_cs_lg']
corpus = ['iclr']

def agregar_nombre(titulo,bolsa):
    columns_names = bolsa.columns.values
    column_name = []
    for v in columns_names:
            column_name.append(titulo+str(v))
    bolsa2 = bolsa.rename(columns=dict(zip(bolsa.columns.values, column_name)))
    print(bolsa2)
    return bolsa2


def colores(X_train,feature_selection):
    GI = ["Think", "Know", "Causal", "Ought", "Perceiv", "Compare", "Eval@", "Eval", "Solve", "Abs@", "ABS", "Quality",
          "Quan", "FREQ", "NUMB", "ORD", "CARD", "DIST", "Time@", "TIME", "Space", "POS", "DIM", "Rel", "COLOR"]
    color=["azul","rojo","verde","cafe","negro","cafe"]
    colores = []
    clasificacio=[]
    palabra=[]
    indice = []
    tcolor={}
    for v in X_train.columns.tolist():
        separacion=v.split("_")
        palabra.append(separacion[1])
        if separacion[1] in GI:
            clasificacio.append("GI")
            colores.append(color[5])
        else:
            if separacion[0] == "TITLE":
                colores.append(color[0])
            if separacion[0] == "ABSTRACT":
                colores.append(color[1])
            if separacion[0] == "INTRODUCTION":
                colores.append(color[2])
            if separacion[0] == "CONCLUSION":
                colores.append(color[3])
            if separacion[0] == "SECTIONS":
                colores.append(color[4])
            clasificacio.append(separacion[0])

    chi2_scores = pd.DataFrame(
        list(zip(clasificacio,palabra, feature_selection.scores_,colores)),
        columns=['clasificacion','ftr', 'score','color'])
    chi2_scores=chi2_scores.sort_values('score', ascending=False)

    score = chi2_scores['score']
    ftr= chi2_scores['ftr']
    clasificacion = chi2_scores['clasificacion']
    coloress = chi2_scores['color']
    tupla2 = list(zip(clasificacion, ftr, coloress))
    i=0
    for b in tupla2:
        i = i + 1
        indice.append(i)
        tcolor[i] = b

    tupla = list(zip(indice,ftr, score))


    f=open("tupla1.json", "w")
    json.dump(tupla,f)
    f.close()

    y = open("diccionario.json", "w")
    json.dump(tcolor, y)
    y.close()

    return chi2_scores

def bow_pos_ngrams_1_3():
    from sklearn.feature_extraction.text import TfidfVectorizer
    secciones = ['TITLE', 'ABSTRACT', 'INTRODUCTION', 'CONCLUSION', 'SECTIONS']
    df = pd.DataFrame()
    for source in corpus:
        df = pd.concat([df, pd.read_csv(source + '_pos_secciones.txt', sep="\t", header=None, on_bad_lines='skip')], axis=0,
                       ignore_index=True)
    df.columns = ["ID", "TIPO", "LABEL", "POS_" + secciones[0], "POS_" + secciones[1], "POS_" + secciones[2],
                  "POS_" + secciones[3], "POS_" + secciones[4]]
    # df["TAGS"] = None
    # print(df)
    df = df.infer_objects()

    # Extrar unicamente el TAG
    for file in range(len(df)):
        for columna in secciones:
            tag = ""
            texto = str(df.loc[file]["POS_" + columna])
            #print(columna)
            #print(texto)
            if texto != "nan" or texto != "":
                texto = texto.split()
                for w in texto:
                    if len(texto) > 1:
                        tag += w.split("|")[0] + " "
                    # print("tag: "+tag)
            df.loc[file, ["TAGS_" + columna]] = tag
    #print(df)
    df["TAGS_ALL"] = df["TAGS_TITLE"] + " " + df["TAGS_ABSTRACT"] + " " + df["TAGS_INTRODUCTION"] + " " + \
                     df["TAGS_CONCLUSION"] + " " + df["TAGS_SECTIONS"]
    #df["TAGS_3"] = df["TAGS_INTRODUCTION"] + " " + df["TAGS_CONCLUSION"] + " " + df["TAGS_SECTIONS"]

    #print(df)
    #pd.options.display.max_columns = None
    #print(df.iloc[0, :])

    # Filtrar data frame
    df_train = df[df['TIPO'] == 'train']
    # df_test = df[df['TIPO'] == 'dev']
    df_test = df[df['TIPO'] == 'test']
    print("Creando bolsa de palabras POS...")

    for columna in secciones:

        text_train = df_train['TAGS_' + columna].values
        text_test = df_test['TAGS_' + columna].values

        # print(text_test)

        vectorizer = TfidfVectorizer(use_idf=False, norm='l2', ngram_range=(1, 3))
        vectorizer.fit(text_train)
        #print(len(vectorizer.vocabulary_))

        X_train = vectorizer.transform(text_train)
        X_test = vectorizer.transform(text_test)

        # print(X_train)

        df_bow_train = pd.DataFrame(X_train.todense(), columns=vectorizer.vocabulary_)
        df_bow_test = pd.DataFrame(X_test.todense(), columns=vectorizer.vocabulary_)
        # print(df_bow_train)
        if columna == 'TITLE':
            df_bow_train_TITLE = df_bow_train
            df_bow_test_TITLE = df_bow_test
        if columna == 'ABSTRACT':
            df_bow_train_ABSTRACT = df_bow_train
            df_bow_test_ABSTRACT = df_bow_test
        if columna == 'INTRODUCTION':
            df_bow_train_INTRODUCTION = df_bow_train
            df_bow_test_INTRODUCTION = df_bow_test
        if columna == 'CONCLUSION':
            df_bow_train_CONCLUSION = df_bow_train
            df_bow_test_CONCLUSION = df_bow_test
        if columna == 'SECTIONS':
            df_bow_train_SECTIONS = df_bow_train
            df_bow_test_SECTIONS = df_bow_test
    text_train = df_train['TAGS_ALL'].values
    text_test = df_test['TAGS_ALL'].values
    # print(text_test)
    vectorizer = TfidfVectorizer(use_idf=False, norm='l2', ngram_range=(1, 3))
    vectorizer.fit(text_train)
    # print(vectorizer.vocabulary_)
    X_train = vectorizer.transform(text_train)
    X_test = vectorizer.transform(text_test)
    # print(X_train)
    print("--- BOLSA POS TITULO ---")
    df_bow_train_TITLE=agregar_nombre("TITLE_",df_bow_train_TITLE)
    print()
    print("--- BOLSA POS RESUMEN ---")
    df_bow_train_ABSTRACT=agregar_nombre("ABSTRACT_",df_bow_train_ABSTRACT)
    print()
    print("--- BOLSA POS INTRODUCCION ---")
    df_bow_train_INTRODUCTION = agregar_nombre("INTRODUCTION_", df_bow_train_INTRODUCTION)
    print()
    print("--- BOLSA POS CONCLUSIÓN ---")
    df_bow_train_CONCLUSION = agregar_nombre("CONCLUSION_", df_bow_train_CONCLUSION)
    print()
    print("--- BOLSA POS CONTENIDO ---")
    df_bow_train_SECTIONS = agregar_nombre("SECTIONS_", df_bow_train_SECTIONS)

    print()
    return df_bow_train_TITLE, df_bow_train_ABSTRACT, df_bow_train_INTRODUCTION, df_bow_train_CONCLUSION, \
           df_bow_train_SECTIONS, df_bow_test_TITLE, df_bow_test_ABSTRACT, df_bow_test_INTRODUCTION, \
           df_bow_test_CONCLUSION, df_bow_test_SECTIONS

def chi_square(X_train, y_train, X_test):
    #nf = sys.argv[1]
    nf =170
    # Select two features with highest chi-squared statistics
    feature_selection = SelectKBest(score_func=chi2, k=nf)
    feature_selection.fit(X_train.values, y_train)
    X_train_fs = feature_selection.transform(X_train.values)
    X_test_fs = feature_selection.transform(X_test.values)
    chi2_scores = colores(X_train, feature_selection)
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None

    # Imprime las 100 mejores caracteristicas
    # print(chi2_scores)
    print(chi2_scores.head(nf))
    print("-------------------")
    # print(kbest)
    with open('Atributos.csv', mode='a', newline='') as archivo:
        # Crea un objeto escritor CSV
        escritor = csv.writer(archivo)
        # Agrega los datos al archivo CSV
        escritor.writerow("------------------")
        escritor.writerow(f"k={nf}")
        chi2_scores.head(nf).to_csv(archivo,header=archivo.tell()==0,index=False)

    #chi2_scores.head(nf).to_csv('Atributos.csv', index=False, header=True)
    return X_train_fs, X_test_fs, nf

#def clasificacion(df, df_bow_pos_train, df_bow_pos_test):
def clasificacion(df, df_bow_train_TITLE, df_bow_train_ABSTRACT, df_bow_train_INTRODUCTION, df_bow_train_CONCLUSION,
                  df_bow_train_SECTIONS, df_bow_test_TITLE, df_bow_test_ABSTRACT, df_bow_test_INTRODUCTION,
                  df_bow_test_CONCLUSION, df_bow_test_SECTIONS):

    from sklearn.metrics import f1_score
    from sklearn import metrics
    # Filtrar data frame
    df_train = df[df['TIPO'] == 'train']
    #df_test = df_RL[df_RL['TIPO'] == 'dev']
    df_test = df[df['TIPO'] == 'test']
    # Obteniendo etiquetas de cada instancia
    y_train = df_train['LABEL'].astype(int)
    y_test = df_test['LABEL'].astype(int)
    # Concatenando con pos
    #150 GRAL INQUIRER
    #30 RIQUEZA LEXICA
    df_train = pd.concat([df_train.iloc[:, 0:155].reset_index(drop=True), df_bow_train_TITLE, df_bow_train_ABSTRACT,
                          df_bow_train_INTRODUCTION, df_bow_train_CONCLUSION, df_bow_train_SECTIONS], axis=1)
    df_test = pd.concat([df_test.iloc[:, 0:155].reset_index(drop=True), df_bow_test_TITLE, df_bow_test_ABSTRACT,
                         df_bow_test_INTRODUCTION, df_bow_test_CONCLUSION, df_bow_test_SECTIONS], axis=1)
    print("--- MATRIZ FINAL ---")
    print(df_train)
    #print(df_test)
    #df_train = df_bow_pos_train
    #df_test = df_bow_pos_test
    # Conversion de tipo de columnas del DataFrame
    df_train = df_train.infer_objects()
    df_test = df_test.infer_objects()
    #ELIGIENDO SOLO ATRIBUTOS ALL
    # 0:6 RIQUEZA LEXICA ALL, 6:24 TITULO Y ABSTRACT SEPARADO DEL RESTO, 12:42 RIQUEZA LEXICA POR SECCIONES
    # 0:25 INQUIRER ALL, 25:100 TITULO Y ABSTRACT SEPARADO DEL RESTO, 50:175 INQUIRER POR SECCIONES
    #df_train = df_train.iloc[:, 50:175]
    #df_test = df_test.iloc[:, 50:175]
    #print(df_train)
    # print(df_test)
    df_test = df_test.astype('float16')
    df_train = df_train.astype('float16')
    print("Sacando principales características...")
    # feature selection
    X_train_fs, X_test_fs,nf = chi_square(df_train, y_train, df_test)
    X_train = X_train_fs
    X_test = X_test_fs
    #  GRAL INQUIRER
    # 0:6 RIQUEZA LEXICA ALL, 6:24 TITULO Y ABSTRACT SEPARADO DEL RESTO, 12:42 RIQUEZA LEXICA POR SECCIONES
    # 0:25 INQUIRER ALL, 25:100 TITULO Y ABSTRACT SEPARADO DEL RESTO, 50:175 INQUIRER POR SECCIONES
    #X_train = df_train.iloc[:, 50:175].values
    #X_test = df_test.iloc[:, 50:175].values

    #X_train = df_train.values
    #X_test = df_test.values

    #print("Sacando principales características por etiqueta...")
    #feature_selection_label(df_train, y_train)

    print("Escalando datos...")
    # Scaling
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    print("Clasificando...")

    # print(X_test)

    # Classification
    from sklearn import svm

    svm_classifier = svm.LinearSVC(class_weight='balanced', C=.012, random_state=0)
    svm_classifier.fit(X_train, y_train)
    # Prediciendo etiquetas
    predictionsSVM = svm_classifier.predict(X_test)
    # Evaluacion
    # scores = cross_val_score(svm_classifier, X_train, y_train, cv=10, scoring="accuracy")
    # print("Metricas cross_validation", scores)
    # print("Media de cross_validation", scores.mean())
    # print("+/-", scores.std())

    scoreSVM = svm_classifier.score(X_test, y_test)
    print()
    cuadro = pd.DataFrame(columns=["SVM Accuracy:","SVM F1:","RF Accuracy:","RF F1:"])
    cuadro = cuadro.append({'k': nf}, ignore_index=True)
    print("SVM Accuracy:", scoreSVM)
    cuadro["SVM Accuracy:"] = (scoreSVM)
    scoreF1_SVM = f1_score(y_test, predictionsSVM, average='macro')
    print("SVM F1:", scoreF1_SVM)
    cuadro["SVM F1:"] = (scoreF1_SVM)
    # Reporte de clasificacion
    reporteSVM = metrics.classification_report(y_test, predictionsSVM)
    print("Reporte de Clasificacion SVM")
    print(reporteSVM)
    # Matriz de confusion
    matriz = metrics.confusion_matrix(y_test, predictionsSVM)
    print("Matriz de confusión SVM")
    print(matriz)
    from sklearn.ensemble import RandomForestClassifier

    # Crear el modelo con 100 arboles
    #rf = RandomForestClassifier(class_weight='balanced', random_state=0)
    rf = RandomForestClassifier(n_estimators=1000, criterion='entropy')
    rf.fit(X_train, y_train)
    # Prediciendo etiquetas
    predictionsRF = rf.predict(X_test)
    # Evaluacion
    # scores = cross_val_score(rf, X_train, y_train, cv=10, scoring="accuracy")
    # print("Metricas cross_validation", scores)
    # print("Media de cross_validation", scores.mean())
    # print("+/-", scores.std())

    scoreRF = rf.score(X_test, y_test)
    print()
    print("RF Accuracy:", scoreRF)
    cuadro["RF Accuracy:"] = (scoreRF)
    scoreF1_RF = f1_score(y_test, predictionsRF, average='macro')
    print("RF F1:", scoreF1_RF)
    cuadro["RF F1:"] = (scoreF1_RF)
    # Reporte de clasificacion
    reporteRF = metrics.classification_report(y_test, predictionsRF)
    print("Reporte de Clasificacion RF")
    print(reporteRF)
    # Matriz de confusion
    matrizRF = metrics.confusion_matrix(y_test, predictionsRF)
    print("Matriz de confusión RF")
    print(matrizRF)
    #cuadro.to_csv('salida.csv')
    with open('salida.csv', mode='a', newline='') as archivo:
        # Crea un objeto escritor CSV
        escritor = csv.writer(archivo)
        # Agrega los datos al archivo CSV
        cuadro.to_csv(archivo, index=False)
    return scoreSVM, scoreF1_SVM, scoreRF, scoreF1_RF

def plot_accuracy(scoreSVM_1, scoreRF_1):
    import matplotlib.pyplot as plt
    import numpy as np
    bow = ['RL + Gral_Inquierer + BOW_POS']
    scores_svm = [round(scoreSVM_1, 3)]
    scores_rf = [round(scoreRF_1, 3)]
    # Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(bow))
    # tamaño de cada barra
    width = 0.35

    fig, ax = plt.subplots()

    # Generamos las barras para svm
    rects1 = ax.bar(x - width / 2, scores_svm, width, label='SVM')
    # Generamos las barras para rf
    rects2 = ax.bar(x + width / 2, scores_rf, width, label='RF')

    # Añadimos las etiquetas de identificacion de valores en el grafico
    ax.set_ylabel('Accuracy')
    ax.set_title('Desempeño (Accuracy)')
    ax.set_xticks(x)
    ax.set_xticklabels(bow)
    # Añadimos un legen() esto permite mmostrar con colores a que pertence cada valor.
    ax.legend()

    # Añadimos las etiquetas para cada barra
    """Funcion para agregar una etiqueta con el valor en cada barra"""
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    """Funcion para agregar una etiqueta con el valor en cada barra"""
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    # Mostramos la grafica con el metodo show()
    #plt.show()

def plot_F1(scoreF1_SVM_1, scoreF1_RF_1):
    import matplotlib.pyplot as plt
    import numpy as np
    bow = ['RL + Gral_Inquierer']
    scores_svm = [round(scoreF1_SVM_1, 3)]
    scores_rf = [round(scoreF1_RF_1, 3)]
    # Obtenemos la posicion de cada etiqueta en el eje de X
    x = np.arange(len(bow))
    # tamaño de cada barra
    width = 0.35

    fig, ax = plt.subplots()

    # Generamos las barras para svm
    rects1 = ax.bar(x - width / 2, scores_svm, width, label='SVM')
    # Generamos las barras para rf
    rects2 = ax.bar(x + width / 2, scores_rf, width, label='RF')

    # Añadimos las etiquetas de identificacion de valores en el grafico
    ax.set_ylabel('F1')
    ax.set_title('Desempeño de POS (F1)')
    ax.set_xticks(x)
    ax.set_xticklabels(bow)
    # Añadimos un legen() esto permite mostrar con colores a que pertence cada valor.
    ax.legend()

    # Añadimos las etiquetas para cada barra
    """Funcion para agregar una etiqueta con el valor en cada barra"""
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    """Funcion para agregar una etiqueta con el valor en cada barra"""
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    # Mostramos la grafica con el metodo show()
    #plt.show()

if __name__ == '__main__':
    print("iniciando")
    df_gral_inquirer = pd.DataFrame()
    df_riqueza_lexica = pd.DataFrame()

    for source in corpus:
        df_riqueza_lexica = pd.concat([df_riqueza_lexica,
                                           pd.read_csv(source+'_riqueza_lexica_ALL.csv')], axis=0, ignore_index=True)
        df_gral_inquirer = pd.concat([df_gral_inquirer,
                                          pd.read_csv(source+'_gral_inquirer_ALL.csv')], axis=0, ignore_index=True)
    #print(df_riqueza_lexica)
    #print(df_gral_inquirer)

    #Concatenando riqueza lexica con gebral inquirer
    df = pd.concat([df_riqueza_lexica.iloc[:, 12:42], df_gral_inquirer.iloc[:, 50:175], df_gral_inquirer.iloc[:, 175:179]], axis=1)

    #df = df_gral_inquirer
    df['LABEL'] = df['LABEL'].astype(int)
    df['ID'] = df['ID'].astype(int)

    #pd.options.display.max_columns = None
    #print(df)

    df_bow_train_TITLE, df_bow_train_ABSTRACT, df_bow_train_INTRODUCTION, df_bow_train_CONCLUSION, \
    df_bow_train_SECTIONS, df_bow_test_TITLE, df_bow_test_ABSTRACT, df_bow_test_INTRODUCTION, df_bow_test_CONCLUSION, \
    df_bow_test_SECTIONS = bow_pos_ngrams_1_3()
    scoreSVM_1, scoreF1_SVM_1, scoreRF_1, scoreF1_RF_1 = clasificacion(df, df_bow_train_TITLE, df_bow_train_ABSTRACT,
                                                                       df_bow_train_INTRODUCTION, df_bow_train_CONCLUSION,
                                                                       df_bow_train_SECTIONS, df_bow_test_TITLE,
                                                                       df_bow_test_ABSTRACT, df_bow_test_INTRODUCTION,
                                                                       df_bow_test_CONCLUSION, df_bow_test_SECTIONS)

    # Graficas
    plot_accuracy(scoreSVM_1, scoreRF_1)
    plot_F1(scoreF1_SVM_1, scoreF1_RF_1)
