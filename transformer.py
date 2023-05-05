import tensorflow as tf
from tensorflow.keras import layers, models

def create_transformer_model(input_shape, d_model, num_heads, num_layers, dff, output_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    for _ in range(num_layers):
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        ffn_output = layers.Dense(dff, activation='relu')(x)
        ffn_output = layers.Dense(d_model)(ffn_output)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(output_dim, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

