# from model.simple_autoencoder import SimpleAutoencoder
# from model.complex_autoencoder import ComplexAutoencoder
# from model.cnn1d_autoencoder import CNN1DAutoencoder
from model.lstm_autoencoder import LSTMAutoencoder


def get_model(model_str: str) -> 'model':
    
    if model_str == 'SimpleAutoencoder':
        return SimpleAutoencoder
    elif model_str == 'ComplexAutoencoder':
        return ComplexAutoencoder
    elif model_str == 'CNN1DAutoencoder':
        return CNN1DAutoencoder
    elif model_str == 'LSTMAutoencoder':
        return LSTMAutoencoder

if __name__ == '__main__':
    pass
