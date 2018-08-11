from keras import Model, Sequential
from keras.layers import LSTM, Lambda, Reshape, Concatenate
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.utils import plot_model

from model.common import WINDOW_TIME_STEPS, MODEL_ARCH_PATH, HEADLINE_SEQ_LEN, EMB_SIZE

HIDDEN_SIZE_HEADLINE = 50
HIDDEN_SIZE_MAIN = 10
N_CONDENSED_OUT = 3
N_NUM = 1  # Only price for now
headline_condenser_g = None


def make_condenser(in_shape, dropout_level=0.3):
    # TODO maybe more outputs?
    condenser = Sequential()
    condenser.add(LSTM(HIDDEN_SIZE_HEADLINE, input_shape=in_shape, activation='relu', name="headline_lstm",
                       dropout=dropout_level))
    condenser.add(Dense((1, N_CONDENSED_OUT), name="headline_dense"))
    # condenser.add(Activation('softmax', name="headline_activation"))
    return condenser


def apply_each(inputs):
    global headline_condenser_g
    outs = list()
    # Split by the time step dimension
    for i in range(inputs.shape[1]):
        outs.append(headline_condenser_g(inputs[:, i, :, :]))

    return Concatenate(axis=1)(outs)


def get_overall_model(headline_len, main_dropout_level=0.4):
    global headline_condenser_g
    headline_in = Input(shape=(WINDOW_TIME_STEPS, headline_len, EMB_SIZE), name="headline_in")
    # Condenser treats each time step of input as separate examples
    headline_condenser_g = make_condenser(in_shape=(headline_len, EMB_SIZE))

    # Apply condenser to each time step of headlines TODO
    headline_out = Lambda(apply_each, name="condense_each_headline")(headline_in)
    # headline_out = TimeDistributed(headline_condenser)(headline_in)
    price_in = Input(shape=(WINDOW_TIME_STEPS, N_NUM), name="price_in")

    # TODO more layers can possibly be added
    x = Concatenate()([headline_out, price_in])  # of shape (WINDOW_SIZE, N_CONCATENATED + N_PRICE)
    x = LSTM(units=HIDDEN_SIZE_MAIN, activation="relu", name="main_lstm")(x)  # TODO maybe another LSTM before this?
    x = Dropout(main_dropout_level, name="main_dropout")(x)
    outputs = Dense(1, name="main_dense")(x)

    model = Model(inputs=[headline_in, price_in], outputs=outputs, name="apple_model")
    return model


if __name__ == '__main__':
    # Build model
    overall_model = get_overall_model(HEADLINE_SEQ_LEN)

    overall_model.summary()
    # Need to have Graphviz in path for this
    print("Plotting...")
    plot_model(overall_model, 'overall.png')
    print("Done.")

    model_json = overall_model.to_json()
    with open(MODEL_ARCH_PATH, 'w') as f:
        f.write(model_json)
