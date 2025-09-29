import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.public import tfxio
import os

# Feature definitions - pastikan ini sama dengan transform module
NUMERICAL_FEATURES = ['Age']
CATEGORICAL_FEATURES = [
    'Disease',
    'Fever',
    'Cough',
    'Fatigue',
    'Difficulty_Breathing',
    'Gender',
    'Blood_Pressure',
    'Cholesterol_Level',
]
LABEL_KEY = 'Outcome_Variable'

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""
    model.tft_layer = tf_transform_output.transform_features_layer()
    # Tentukan input signature
    input_signature = [tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')]
    @tf.function(input_signature=input_signature)
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    return serve_tf_examples_fn.get_concrete_function()

def _get_transform_features_signature(model, tf_transform_output):
    """Gets a signature for transforming features."""
    model.tft_layer = tf_transform_output.transform_features_layer()
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()
    input_signature = [
        {
            key: tf.TensorSpec(
                shape=(None, 1) if spec.dtype == tf.float32 else (None, 1),
                dtype=spec.dtype,
                name=key
            )
            for key, spec in transformed_feature_spec.items()
            if key != LABEL_KEY
        }
    ]
    @tf.function(input_signature=input_signature)
    def transform_features_fn(features):
        transformed_features = model.tft_layer(features)
        return model(transformed_features)
    return transform_features_fn.get_concrete_function()


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=32):
    """Loads TFRecord dataset for training/eval."""
    try:
        transformed_feature_spec = tf_transform_output.transformed_feature_spec()
        
        dataset = data_accessor.tf_dataset_factory(
            file_pattern,
            tfxio.TensorFlowDatasetOptions(
                batch_size=batch_size,
                label_key=LABEL_KEY,
                shuffle=True,
                shuffle_buffer_size=1000
            ),
            schema=tf_transform_output.transformed_metadata.schema
        )
        
        return dataset.repeat()
        
    except Exception as e:
        print(f"Error in _input_fn: {e}")
        raise

def _build_keras_model(tf_transform_output):
    """Build and compile Keras model."""
    
    # Get transformed feature spec
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()
    
    print(f"Available transformed features: {list(transformed_feature_spec.keys())}")
    
    # Input layers untuk semua fitur yang sudah di-transform
    inputs = {}
    
    # Handle numerical features - sesuaikan dengan transform output
    numerical_inputs = []
    for feature in NUMERICAL_FEATURES:
        # Nama yang seharusnya ada berdasarkan transform module
        transformed_name = f"{feature}"
        
        if transformed_name in transformed_feature_spec:
            inputs[transformed_name] = keras.layers.Input(
                shape=(1,), name=transformed_name, dtype=tf.float32
            )
            numerical_inputs.append(inputs[transformed_name])
            print(f"✓ Found numerical feature: {transformed_name}")
        else:
            print(f"⚠️  Expected numerical feature '{transformed_name}' not found")
            print(f"Available features: {list(transformed_feature_spec.keys())}")
    
    # Handle categorical features - sesuaikan dengan transform output
    embedded = []
    for feature in CATEGORICAL_FEATURES:
        # Nama yang seharusnya ada berdasarkan transform module
        transformed_name = f"{feature}"
        
        if transformed_name in transformed_feature_spec:
            # Coba dapatkan vocab size dengan berbagai cara
            vocab_size = None
            vocab_name = f"{feature}_vocab"  # Sesuai dengan transform module
            
            try:
                vocab_size = tf_transform_output.vocabulary_size_by_name(vocab_name)
                print(f"✓ Found vocab for '{feature}': actual size={vocab_size}")
            except ValueError as e:
                print(f"⚠️  Cannot get vocab size for '{vocab_name}': {e}")
                
                # Fallback: inspect the transformed feature spec untuk estimasi
                try:
                    feature_spec = transformed_feature_spec[transformed_name]
                    print(f"Feature spec for {transformed_name}: {feature_spec}")
                    
                    # Jika ada info shape, gunakan untuk estimasi
                    vocab_size = 50  # Conservative fallback
                    print(f"Using conservative fallback vocab size: {vocab_size}")
                except:
                    vocab_size = 50  # Safe fallback
            
            # PENTING: Tambahkan buffer yang cukup untuk OOV dan unexpected values
            # TFT biasanya menyimpan: 0=padding, vocab_values, then OOV buckets
            embedding_vocab_size = vocab_size + 10  # Tambah buffer yang lebih besar
            
            print(f"Setting embedding vocab size for '{feature}': {embedding_vocab_size}")
            
            # Buat input layer dan embedding
            inputs[transformed_name] = keras.layers.Input(
                shape=(1,), name=transformed_name, dtype=tf.int64
            )
            
            embed_dim = min(50, max(4, vocab_size // 2))
            x = keras.layers.Embedding(
                input_dim=embedding_vocab_size,  # Dengan buffer yang aman
                output_dim=embed_dim,
                mask_zero=False  # Tidak menggunakan mask untuk menghindari kompleksitas
            )(inputs[transformed_name])
            x = keras.layers.Reshape((embed_dim,))(x)
            embedded.append(x)
            
        else:
            print(f"⚠️  Expected categorical feature '{transformed_name}' not found")
            print(f"Available features: {list(transformed_feature_spec.keys())}")
    
    # Concatenate all inputs
    all_inputs = numerical_inputs + embedded
    
    if len(all_inputs) == 0:
        raise ValueError(f"No valid features found for model input. Available features: {list(transformed_feature_spec.keys())}")
    elif len(all_inputs) == 1:
        concatenated = all_inputs[0]
    else:
        concatenated = keras.layers.concatenate(all_inputs)
    
    print(f"✓ Model input created with {len(all_inputs)} feature groups")
    
    # Deep neural network layers
    x = keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-5))(concatenated)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.15)(x)
    
    x = keras.layers.Dense(16, activation='relu')(x)
    
    # Output layer untuk binary classification
    outputs = keras.layers.Dense(1, activation='sigmoid', name='predictions')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def run_fn(fn_args: FnArgs):
    """Main training function that TFX will call."""
    
    print("=" * 50)
    print("Starting TFX Trainer...")
    print(f"Transform output: {fn_args.transform_output}")
    print(f"Train files: {fn_args.train_files}")
    print(f"Eval files: {fn_args.eval_files}")
    print(f"Serving model dir: {fn_args.serving_model_dir}")
    print("=" * 50)
    
    try:
        # Load TF Transform output
        tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
        print("✓ TF Transform output loaded successfully")
        
        # Create datasets
        train_dataset = _input_fn(
            fn_args.train_files, 
            fn_args.data_accessor, 
            tf_transform_output, 
            batch_size=32
        )
        print("✓ Training dataset created")
        
        eval_dataset = _input_fn(
            fn_args.eval_files, 
            fn_args.data_accessor, 
            tf_transform_output, 
            batch_size=32
        )
        print("✓ Evaluation dataset created")
        
        # Build model
        model = _build_keras_model(tf_transform_output)
        print("✓ Model built successfully")
        print("\nModel Summary:")
        model.summary()
        
        # Training configuration
        train_steps = fn_args.train_steps or 1000
        eval_steps = fn_args.eval_steps or 200
        
        # Calculate epochs and steps
        steps_per_epoch = min(100, max(10, train_steps // 10))
        epochs = max(1, train_steps // steps_per_epoch)
        validation_steps = min(50, max(5, eval_steps // 2))
        
        print(f"\nTraining Configuration:")
        print(f"Total train steps: {train_steps}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Epochs: {epochs}")
        print(f"Validation steps: {validation_steps}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=3,
                verbose=1,
                min_lr=1e-7
            )
        ]
        
        # Train the model
        print("\nStarting training...")
        history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=eval_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        print("✓ Training completed successfully")
        
        # Create serving signatures
        signatures = {
            'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output),
            'transform_features': _get_transform_features_signature(model, tf_transform_output)
        }
        
        # Save the model
        print(f"\nSaving model to: {fn_args.serving_model_dir}")
        model.save(
            fn_args.serving_model_dir, 
            save_format='tf', 
            signatures=signatures
        )
        print("✓ Model saved successfully!")
        
        # Print training history summary
        if history and history.history:
            final_loss = history.history.get('loss', [])[-1] if history.history.get('loss') else 'N/A'
            final_accuracy = history.history.get('accuracy', [])[-1] if history.history.get('accuracy') else 'N/A'
            final_val_loss = history.history.get('val_loss', [])[-1] if history.history.get('val_loss') else 'N/A'
            final_val_accuracy = history.history.get('val_accuracy', [])[-1] if history.history.get('val_accuracy') else 'N/A'
            
            print(f"\nFinal Training Metrics:")
            print(f"Loss: {final_loss}")
            print(f"Accuracy: {final_accuracy}")
            print(f"Val Loss: {final_val_loss}")
            print(f"Val Accuracy: {final_val_accuracy}")
        
        print("=" * 50)
        print("TFX Trainer completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error in training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
