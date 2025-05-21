import tensorflow as tf

def build_model(X_train, num_classes):
    # Définir le modèle
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),       
        tf.keras.layers.Dense(64, activation='relu'),  # Je remplace 'sigmoid' par 'relu' ici pour une meilleure convergence
        tf.keras.layers.Dense(1, activation='sigmoid')  # Pour la classification binaire, une sortie avec activation sigmoid
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00012589254117941674),  # Taux d'apprentissage adapté
        loss='binary_crossentropy',  # Perte pour la classification binaire
        metrics=['recall', tf.keras.metrics.AUC(name="auc")]  # Mesures de recall et AUC
    )

    model.summary()  # Résumé du modèle pour vérification
    return model
