# coding: utf-8
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from loss import org_mse
from import_data import make_data
from model import make_model
from evaluate_model import evaluate

BATCH_SIZE = 4
EPOCH = 1


def main():

    (x_train, x_test, y_xywh_train, y_xywh_test,
     y_c_train, y_c_test, z_train, z_test) = make_data()

    (inputs, main_output, side_output, numbers_output) = make_model()

    model = Model(inputs=inputs, outputs=[
                  main_output, side_output, numbers_output])

    adam = Adam(lr=0.0003)
    checkpointer = ModelCheckpoint("model1.h5", monitor='val_loss', verbose=0,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto', period=1)

    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=190, verbose=0, mode='auto')

    model.compile(
        optimizer=adam,
        loss={
            'main_output': org_mse,
            'side_output': 'binary_crossentropy',
            'numbers_output': 'categorical_crossentropy'
        },
        metrics={
            'main_output': 'mae',
            'side_output': 'accuracy',
            'numbers_output': 'accuracy'
        },
        loss_weights={
            'main_output': 1.,
            'side_output': 0.2,
            'numbers_output': 1
        })

    model.fit(x=x_train, y=[y_xywh_train, y_c_train, z_train],
              batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1,
              validation_split=0.1,
              callbacks=[checkpointer, early_stopping])

    score = model.evaluate(
        x_test, [y_xywh_test, y_c_test, z_test], verbose=1,
        batch_size=BATCH_SIZE)

    print(score[0], score[1])

    json_string = model.to_json()
    with open('model.json', 'w') as f:
        f.write(json_string)

        model.save_weights('param.hdf5')
    model.save('model.h5')

    evaluate()

if __name__ == "__main__":
    main()
