import tensorflow as tf

def build_model(X_train, num_classes):
    # Réseau avec 2 couches cachées de 64 neurones chacune
    # et une couche de sortie avec activation softmax pour classification
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Définition de la fonction de perte, de l'optimiseur et des métriques
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy',tf.keras.metrics.AUC(name = "auc")]
    )

    model.summary()

    return model