import tensorflow as tf

def build_model(X_train, num_classes):
    """
    Construit et compile un modèle de réseau de neurones pour une tâche de classification binaire.

    Paramètres
    ----------
    X_train : numpy.ndarray
        Données d'entraînement, utilisées pour définir la forme de l'entrée du modèle.
    num_classes : int
        Nombre de classes cibles. (Ce paramètre n'est pas utilisé dans la version actuelle du modèle.)

    Retourne
    -------
    tensorflow.keras.Model
        Un modèle Keras compilé, prêt à être entraîné.

    Notes
    -----
    - Le modèle est un réseau de neurones séquentiel composé de :
        - Une couche d'entrée correspondant à la dimension des features de `X_train`
        - Deux couches cachées (8 et 64 neurones) avec activation ReLU
        - Une couche de sortie avec une activation sigmoïde pour une sortie binaire
    - L’optimiseur utilisé est Adam avec un taux d’apprentissage fixé à ~1.26e-4.
    - La fonction de perte est `binary_crossentropy`, adaptée à la classification binaire.
    - Les métriques utilisées pour l’évaluation sont le `recall` et l’AUC (aire sous la courbe ROC).
    - Le résumé du modèle est affiché via `model.summary()`.
    """

    # Définir le modèle
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),       
        tf.keras.layers.Dense(8, activation='sigmoid'),  # Je remplace 'sigmoid' par 'relu' ici pour une meilleure convergence
        tf.keras.layers.Dense(num_classes, activation='sigmoid')  # Pour la classification binaire, une sortie avec activation sigmoid
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00012589254117941674),  # Taux d'apprentissage adapté
        loss='binary_crossentropy',  # Perte pour la classification binaire
        metrics=['recall', tf.keras.metrics.AUC(name="auc")]  # Mesures de recall et AUC
    )

    model.summary()  # Résumé du modèle pour vérification
    return model
