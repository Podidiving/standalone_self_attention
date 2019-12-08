from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback

def get_callbacks(class_names):
    num_classes = len(class_names)
    return [
        AccuracyCallback(num_classes=num_classes),
        AUCCallback(
            num_classes=num_classes,
            input_key="targets_one_hot",
            class_names=class_names
        ),
        F1ScoreCallback(
            input_key="targets_one_hot",
            activation="Softmax"
        )
    ]
