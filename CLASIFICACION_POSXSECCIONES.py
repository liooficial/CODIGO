import pandas as pd

#corpus = ['iclr', 'acl', 'arxiv_cs_ai', 'arxiv_cs_cl', 'arxiv_cs_lg']
corpus = ['arxiv_cs_ai', 'arxiv_cs_cl', 'arxiv_cs_lg']

def clasificacion(df):
    from sklearn.metrics import f1_score
    from sklearn import metrics

    # Filtrar data frame
    df_train = df[df['TIPO'] == 'train']
    #df_test = df_RL[df_RL['TIPO'] == 'dev']
    df_test = df[df['TIPO'] == 'test']

    # Obteniendo etiquetas de cada instancia
    y_train = df_train['LABEL'].astype(int)
    y_test = df_test['LABEL'].astype(int)
    print(df_train)

    df_train = df_train.iloc[:, 0:30].reset_index(drop=True)
    df_test = df_test.iloc[:, 0:30].reset_index(drop=True)


    # Conversion de tipo de columnas del DataFrame
    df_train = df_train.infer_objects()
    df_test = df_test.infer_objects()

    X_train = df_train.values
    X_test = df_test.values

    print("Escalando datos...")
    # Scaling
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    print("Clasificando...")

    # Classification
    from sklearn import svm

    svm_classifier = svm.LinearSVC(class_weight='balanced', C=.012, random_state=0)
    svm_classifier.fit(X_train, y_train)
    # Prediciendo etiquetas
    predictionsSVM = svm_classifier.predict(X_test)
    # Evaluacion
    scoreSVM = svm_classifier.score(X_test, y_test)
    print()
    print("SVM Accuracy:", scoreSVM)
    scoreF1_SVM = f1_score(y_test, predictionsSVM, average='macro')
    print("SVM F1:", scoreF1_SVM)
    # Reporte de clasificacion
    reporteSVM = metrics.classification_report(y_test, predictionsSVM)
    print("Reporte de Clasificacion SVM")
    print(reporteSVM)
    # Matriz de confusion
    matriz = metrics.confusion_matrix(y_test, predictionsSVM)
    print("Matriz de confusión SVM")
    print(matriz);

    from sklearn.ensemble import RandomForestClassifier

    # Crear el modelo con 100 arboles
    #rf = RandomForestClassifier(class_weight='balanced', random_state=0)
    rf = RandomForestClassifier(n_estimators=1000, criterion='entropy')
    rf.fit(X_train, y_train)
    # Prediciendo etiquetas
    predictionsRF = rf.predict(X_test)
    # Evaluacion
    scoreRF = rf.score(X_test, y_test)
    print()
    print("RF Accuracy:", scoreRF)
    scoreF1_RF = f1_score(y_test, predictionsRF, average='macro')
    print("RF F1:", scoreF1_RF)
    # Reporte de clasificacion
    reporteRF = metrics.classification_report(y_test, predictionsRF)
    print("Reporte de Clasificacion RF")
    print(reporteRF)
    # Matriz de confusion
    matrizRF = metrics.confusion_matrix(y_test, predictionsRF)
    print("Matriz de confusión RF")
    print(matrizRF);

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
    plt.show()

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
    plt.show()

if __name__ == '__main__':
    df_riqueza_lexica = pd.DataFrame()
    for source in corpus:
        df_riqueza_lexica = pd.concat([df_riqueza_lexica,
                                           pd.read_csv(source+'_riqueza_lexica.csv')], axis=0, ignore_index=True)


    df = df_riqueza_lexica
    df['LABEL'] = df['LABEL'].astype(int)
    df['ID'] = df['ID'].astype(int)
    #print(df)
    #pd.options.display.max_columns = None
    #print(df)

    scoreSVM_1, scoreF1_SVM_1, scoreRF_1, scoreF1_RF_1 = clasificacion(df)
    #scoreSVM_1, scoreF1_SVM_1, scoreRF_1, scoreF1_RF_1 = clasificacion(df=df)

    # Graficas
    plot_accuracy(scoreSVM_1, scoreRF_1)
    plot_F1(scoreF1_SVM_1, scoreF1_RF_1)