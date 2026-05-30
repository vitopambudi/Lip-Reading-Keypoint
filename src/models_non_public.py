"""
Module: models.py
Description: Module for build all models that use in the experiment
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, 
    Dropout, BatchNormalization, Bidirectional, GRU, LSTM
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW

class SequenceModelFactory:
    
    @staticmethod
    def build_model(model_type: str, input_shape: tuple, num_classes: int) -> Model:
        
        inputs = Input(shape=input_shape, name="input_layer")
        model_type = model_type.upper()

        # ==========================================
        # PENENTUAN HIPERPARAMETER SPESIFIK MODEL
        # ==========================================
        if model_type == '1DCNN':
            k_sizes = [3, 3, 3]
            l2_reg = l2(5e-4)
            cnn_drops = [0.3, 0.3, 0.4]
            k_init_1 = 'glorot_uniform'
            
        elif model_type in ['1DCNN-LSTM', '1DCNN-GRU', '1DCNN-BIGRU']:
            # Ketiga model ini berbagi fondasi CNN yang identik
            k_sizes = [5, 5, 3]
            l2_reg = l2(1e-3)
            cnn_drops = [0.3, 0.35, 0.35]
            k_init_1 = 'glorot_uniform'
            
        elif model_type == '1DCNN-BILSTM':
            k_sizes = [5, 5, 3]
            l2_reg = l2(5e-4)
            cnn_drops = [0.25, 0.3, 0.3]
            k_init_1 = 'he_normal'
            
        else:
            raise ValueError(
                f"Model type '{model_type}' is unknown. Choose one of these models: "
                "['1DCNN', '1DCNN-LSTM', '1DCNN-BILSTM', '1DCNN-GRU', '1DCNN-BIGRU']"
            )

        # ==========================================
        # Block 1: Spatial Features Extraction with 1DCNN
        # ==========================================
        x = Conv1D(64, kernel_size=k_sizes[0], activation='relu', padding='same', 
                   kernel_initializer=k_init_1, kernel_regularizer=l2_reg, name="conv1d_1")(inputs)
        x = BatchNormalization(name="batchnorm_1")(x)
        x = MaxPooling1D(pool_size=2, name="maxpool_1")(x)
        x = Dropout(cnn_drops[0], name="dropout_1")(x)

        x = Conv1D(128, kernel_size=k_sizes[1], activation='relu', padding='same', 
                   kernel_regularizer=l2_reg, name="conv1d_2")(x)
        x = BatchNormalization(name="batchnorm_2")(x)
        x = MaxPooling1D(pool_size=2, name="maxpool_2")(x)
        x = Dropout(cnn_drops[1], name="dropout_2")(x)

        x = Conv1D(256, kernel_size=k_sizes[2], activation='relu', padding='same', 
                   kernel_regularizer=l2_reg, name="conv1d_3")(x)
        x = BatchNormalization(name="batchnorm_3")(x)
        x = MaxPooling1D(pool_size=2, name="maxpool_3")(x)
        x = Dropout(cnn_drops[2], name="dropout_3")(x)

        # ==========================================
        # Block 2: TEMPORAL & FULLY CONNECTED
        # ==========================================
        if model_type == '1DCNN':
            x = Flatten(name="flatten_temporal")(x)
            x = Dense(128, activation='relu', kernel_regularizer=l2_reg, name="dense_1")(x)
            x = BatchNormalization(name="batchnorm_fc")(x) 
            x = Dropout(0.5, name="dropout_fc")(x)
            
        elif model_type == '1DCNN-LSTM':
            x = LSTM(128, return_sequences=False, recurrent_dropout=0.2, name="lstm_temporal")(x)
            x = Dropout(0.4, name="dropout_lstm")(x)
            x = Dense(128, activation='relu', kernel_regularizer=l2_reg, name="dense_1")(x)
            x = Dropout(0.5, name="dropout_fc")(x)
            
        elif model_type == '1DCNN-BILSTM':
            x = Bidirectional(
                LSTM(128, return_sequences=False), 
                name="bilstm_temporal"
            )(x)
            x = Dropout(0.4, name="dropout_bilstm")(x)
            x = Dense(128, activation='relu', kernel_regularizer=l2_reg, name="dense_1")(x)
            x = Dropout(0.5, name="dropout_fc")(x)
            
        elif model_type == '1DCNN-GRU':
            x = GRU(128, return_sequences=False, recurrent_dropout=0.2, name="gru_temporal")(x)
            x = Dropout(0.4, name="dropout_gru")(x)
            x = Dense(128, activation='relu', kernel_regularizer=l2_reg, name="dense_1")(x)
            x = Dropout(0.5, name="dropout_fc")(x)
            
        elif model_type == '1DCNN-BIGRU':
            x = Bidirectional(
                GRU(128, return_sequences=False, recurrent_dropout=0.2), 
                name="bigru_temporal"
            )(x)
            x = Dropout(0.4, name="dropout_bigru")(x)
            x = Dense(128, activation='relu', kernel_regularizer=l2_reg, name="dense_1")(x)
            x = Dropout(0.5, name="dropout_fc")(x) 

        # ==========================================
        # Block 3: OUTPUT LAYER
        # ==========================================
        outputs = Dense(num_classes, activation='softmax', name="output_layer")(x)

        return Model(inputs=inputs, outputs=outputs, name=model_type)

    @staticmethod
    def compile_model(model: Model, learning_rate: float = 1e-3, weight_decay: float = 1e-5) -> Model:
        model.compile(
            optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model