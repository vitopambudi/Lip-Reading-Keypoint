"""
Module: models.py
Description: Model Factory for the baseline 1DCNN architecture.
Note: Advanced hybrid architectures (LSTM, biLSTM, GRU, biGRU) are temporarily 
      hidden to protect intellectual property and for scientific publication purposes.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, 
    Dropout, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW

class SequenceModelFactory:
    
    @staticmethod
    def build_model(model_type: str, input_shape: tuple, num_classes: int) -> Model:
        """
        Builds the model based on the selected architecture type.
        The baseline model (pure 1DCNN) is open-sourced for pipeline testing.
        Hybrid models are intentionally locked for research protection.
        """
        inputs = Input(shape=input_shape, name="input_layer")
        model_type = model_type.upper()

        # ==========================================
        # MODEL RULES & HYPERPARAMETER CONFIGURATION
        # ==========================================
        if model_type == '1DCNN':
            k_sizes = [3, 3, 3]
            l2_reg = l2(5e-4)
            cnn_drops = [0.3, 0.3, 0.4]
            k_init_1 = 'glorot_uniform'
            
        elif model_type in ['1DCNN-LSTM', '1DCNN-GRU', '1DCNN-BIGRU', '1DCNN-BILSTM']:
            # Professional exception to protect active research IP
            raise PermissionError(
                f"Access to the '{model_type}' architecture is temporarily suspended. "
                "This hybrid architecture is currently under review for journal publication and IP registration. "
                "Please run the script with the '--model 1DCNN' parameter to verify pipeline functionality."
            )
            
        else:
            raise ValueError(
                f"Model type '{model_type}' is not recognized. Available types: ['1DCNN']"
            )

        # ==========================================
        # BLOCK 1: SPATIAL FEATURE EXTRACTION (1D-CNN)
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
        # BLOCK 2: BASELINE CLASS-HEAD (1DCNN)
        # ==========================================
        x = Flatten(name="flatten_temporal")(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2_reg, name="dense_1")(x)
        x = BatchNormalization(name="batchnorm_fc")(x) 
        x = Dropout(0.5, name="dropout_fc")(x)

        # ==========================================
        # BLOCK 3: OUTPUT LAYER
        # ==========================================
        outputs = Dense(num_classes, activation='softmax', name="output_layer")(x)

        return Model(inputs=inputs, outputs=outputs, name=model_type)

    @staticmethod
    def compile_model(model: Model, learning_rate: float = 1e-3, weight_decay: float = 1e-5) -> Model:
        """
        Compiles the initialized model using the AdamW optimizer.
        """
        model.compile(
            optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model