import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
from tensorflow_transform.tf_metadata import schema_utils
from tfx_bsl.public import tfxio
import kerastuner as kt
from typing import Dict, Any, List
import os

# Feature definitions - sama dengan trainer module
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

def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=32):
    """Loads TFRecord dataset for training/eval - sama dengan trainer."""
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

def _build_keras_model_with_hyperparams(tf_transform_output, hparams):
    """Build Keras model with tunable hyperparameters."""
    
    # Get transformed feature spec
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()
    print(f"Available transformed features: {list(transformed_feature_spec.keys())}")
    
    # Input layers untuk semua fitur yang sudah di-transform
    inputs = {}
    
    # Handle numerical features
    numerical_inputs = []
    for feature in NUMERICAL_FEATURES:
        transformed_name = f"{feature}"
        
        if transformed_name in transformed_feature_spec:
            inputs[transformed_name] = keras.layers.Input(
                shape=(1,), name=transformed_name, dtype=tf.float32
            )
            numerical_inputs.append(inputs[transformed_name])
            print(f"✓ Found numerical feature: {transformed_name}")
    
    # Handle categorical features dengan tunable embedding dimensions
    embedded = []
    for feature in CATEGORICAL_FEATURES:
        transformed_name = f"{feature}"
        
        if transformed_name in transformed_feature_spec:
            # Get vocabulary size
            vocab_size = None
            vocab_name = f"{feature}_vocab"
            
            try:
                vocab_size = tf_transform_output.vocabulary_size_by_name(vocab_name)
                print(f"✓ Found vocab for '{feature}': actual size={vocab_size}")
            except ValueError as e:
                print(f"⚠️  Cannot get vocab size for '{vocab_name}': {e}")
                vocab_size = 50  # Conservative fallback
            
            # Buffer untuk OOV values
            embedding_vocab_size = vocab_size + 10
            
            # Tunable embedding dimension
            embed_dim = hparams.Int(
                f'embed_dim_{feature}',
                min_value=4,
                max_value=min(64, max(8, vocab_size // 2)),
                default=min(32, max(4, vocab_size // 4))
            )
            
            print(f"Setting embedding for '{feature}': vocab_size={embedding_vocab_size}, embed_dim={embed_dim}")
            
            # Create input layer dan embedding
            inputs[transformed_name] = keras.layers.Input(
                shape=(1,), name=transformed_name, dtype=tf.int64
            )
            
            x = keras.layers.Embedding(
                input_dim=embedding_vocab_size,
                output_dim=embed_dim,
                mask_zero=False
            )(inputs[transformed_name])
            x = keras.layers.Reshape((embed_dim,))(x)
            embedded.append(x)
    
    # Concatenate all inputs
    all_inputs = numerical_inputs + embedded
    
    if len(all_inputs) == 0:
        raise ValueError(f"No valid features found for model input. Available: {list(transformed_feature_spec.keys())}")
    elif len(all_inputs) == 1:
        concatenated = all_inputs[0]
    else:
        concatenated = keras.layers.concatenate(all_inputs)
    
    print(f"✓ Model input created with {len(all_inputs)} feature groups")
    
    # Tunable architecture parameters
    num_layers = hparams.Int('num_layers', min_value=2, max_value=5, default=3)
    use_batch_norm = hparams.Boolean('use_batch_norm', default=True)
    use_dropout = hparams.Boolean('use_dropout', default=True)
    activation_type = hparams.Choice('activation', ['relu', 'leaky_relu', 'elu'], default='leaky_relu')
    
    # Build tunable deep neural network
    x = concatenated
    
    for i in range(num_layers):
        # Tunable layer sizes - decreasing pattern
        if i == 0:  # First layer
            units = hparams.Int('units_first', min_value=64, max_value=256, step=32, default=128)
        elif i == num_layers - 1:  # Last layer
            units = hparams.Int('units_last', min_value=16, max_value=64, step=8, default=32)
        else:  # Middle layers
            units = hparams.Int(f'units_middle_{i}', min_value=32, max_value=128, step=16, default=64)
        
        # Dense layer dengan tunable L2 regularization
        l2_reg = hparams.Float('l2_regularization', min_value=1e-6, max_value=1e-3, sampling='log', default=1e-5)
        x = keras.layers.Dense(units, kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        
        # Optional batch normalization
        if use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        
        # Tunable activation
        if activation_type == 'leaky_relu':
            alpha = hparams.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, default=0.1)
            x = keras.layers.LeakyReLU(alpha=alpha)(x)
        elif activation_type == 'elu':
            x = keras.layers.ELU()(x)
        else:  # relu
            x = keras.layers.ReLU()(x)
        
        # Optional dropout dengan tunable rate
        if use_dropout and i < num_layers - 1:  # No dropout on last layer
            dropout_rate = hparams.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, default=0.2)
            x = keras.layers.Dropout(dropout_rate)(x)
    
    # Final dense layer sebelum output
    final_units = hparams.Int('final_units', min_value=8, max_value=32, step=4, default=16)
    x = keras.layers.Dense(final_units, activation='relu')(x)
    
    # Output layer untuk binary classification
    outputs = keras.layers.Dense(1, activation='sigmoid', name='predictions')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Tunable optimizer dan learning rate
    learning_rate = hparams.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-3)
    optimizer_type = hparams.Choice('optimizer', ['adam', 'rmsprop'], default='adam')
    
    if optimizer_type == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    # elif optimizer_type == 'adamw':
    #     weight_decay = hparams.Float('weight_decay', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-4)
    #     optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:  # rmsprop
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    print(f"✓ Model built with tunable hyperparameters:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Optimizer: {optimizer_type}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Batch norm: {use_batch_norm}")
    print(f"  - Dropout: {use_dropout}")
    
    return model
 
def tuner_fn(fn_args) -> TunerFnResult:
    """TFX Tuner function untuk hyperparameter optimization."""
    
    print("=" * 60)
    print("Starting TFX Tuner for Medical Diagnosis Model...")
    print(f"Transform output: {fn_args.transform_graph_path}")
    print(f"Train files: {fn_args.train_files}")
    print(f"Eval files: {fn_args.eval_files}")
    print(f"Working directory: {fn_args.working_dir}")
    print("=" * 60)
    
    try:
        # Load TF Transform output
        tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
        print("✓ TF Transform output loaded successfully")
        
        # Create datasets untuk tuning
        train_dataset = _input_fn(
            fn_args.train_files[0],
            fn_args.data_accessor,
            tf_transform_output,
            batch_size=32
        )
        print("✓ Training dataset created")
        
        eval_dataset = _input_fn(
            fn_args.eval_files[0],
            fn_args.data_accessor,
            tf_transform_output,
            batch_size=32
        )
        print("✓ Evaluation dataset created")
        
        # Hypermodel function
        def build_model(hp):
            return _build_keras_model_with_hyperparams(tf_transform_output, hp)
        
        # Setup tuner - menggunakan Bayesian Optimization untuk efisiensi
        tuner = kt.BayesianOptimization(
            hypermodel=build_model,
            objective=kt.Objective('val_auc', direction='max'),  # Optimize AUC untuk medical diagnosis
            max_trials=15,  # Reasonable number untuk medical model
            num_initial_points=3,
            directory=fn_args.working_dir,
            project_name='medical_diagnosis_tuning',
            seed=42,
            overwrite=True
        )
        
        print("✓ Bayesian Optimization tuner initialized")
        print(f"  - Max trials: 15")
        print(f"  - Objective: Maximize validation AUC")
        print(f"  - Initial random points: 3")
        
        # Calculate training parameters
        train_steps = fn_args.train_steps or 1000
        eval_steps = fn_args.eval_steps or 200
        
        # For tuning, use shorter training to speed up search
        tuning_epochs = 15  # Shorter untuk tuning
        steps_per_epoch = min(50, max(10, train_steps // 20))  # Fewer steps per epoch
        validation_steps = min(25, max(5, eval_steps // 4))
        
        print(f"\nTuning Configuration:")
        print(f"Training epochs per trial: {tuning_epochs}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Enhanced callbacks untuk tuning
        def get_callbacks():
            return [
                # Early stopping untuk avoid overfitting during tuning
                keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=5,
                    restore_best_weights=True,
                    verbose=0,
                    mode='max'
                ),
                
                # Reduce learning rate untuk fine-tuning
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=3,
                    verbose=0,
                    min_lr=1e-7
                ),
                
                # Stop unpromising trials early
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=7,
                    restore_best_weights=True,
                    verbose=0,
                    mode='min'
                )
            ]
        
        # Return TunerFnResult
        
        return TunerFnResult(
            tuner=tuner,
            fit_kwargs={
                'x': train_dataset,
                'validation_data': eval_dataset,
                'steps_per_epoch': steps_per_epoch,
                'validation_steps': validation_steps,
                'epochs': tuning_epochs,
                'callbacks': get_callbacks(),
                'verbose': 1
            }
        )
        
    except Exception as e:
        print(f"❌ Error in tuner setup: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# # Advanced tuner function dengan lebih banyak strategi
# def advanced_tuner_fn(fn_args: FnArgs):
#     """Advanced tuner dengan multiple objectives dan strategi."""
    
#     print("=" * 60)
#     print("Starting ADVANCED TFX Tuner for Medical Diagnosis...")
#     print("=" * 60)
    
#     try:
#         tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
        
#         # Datasets
#         train_dataset = _input_fn(
#             fn_args.train_files, 
#             fn_args.data_accessor, 
#             tf_transform_output, 
#             batch_size=32
#         )
#         print("✓ Training dataset created")
        
#         eval_dataset = _input_fn(
#             fn_args.eval_files, 
#             fn_args.data_accessor, 
#             tf_transform_output, 
#             batch_size=32
#         )
#         print("✓ Evaluation dataset created")
        
#         # Advanced hypermodel dengan conditional parameters
#         def advanced_build_model(hp):
#             model = _build_keras_model_with_hyperparams(tf_transform_output, hp)
            
#             # Advanced tuning: conditional architectures
#             use_residual = hp.Boolean('use_residual_connections', default=False)
            
#             if use_residual:
#                 print("🔧 Using residual connections in model")
#                 # Implementasi residual connections bisa ditambahkan di sini
            
#             # Class weight balancing untuk medical data
#             use_class_weights = hp.Boolean('use_class_weights', default=True)
#             if use_class_weights:
#                 # Untuk medical diagnosis, biasanya imbalanced
#                 class_weight = {
#                     0: hp.Float('class_weight_negative', 0.3, 1.0, default=0.5),
#                     1: hp.Float('class_weight_positive', 1.0, 3.0, default=1.5)
#                 }
#                 print(f"🔧 Using class weights: {class_weight}")
            
#             return model
        
#         # Multi-objective tuner
#         tuner = kt.BayesianOptimization(
#             hypermodel=advanced_build_model,
#             objective=[
#                 kt.Objective('val_auc', direction='max'),           # Primary: AUC
#                 kt.Objective('val_precision', direction='max'),     # Secondary: Precision
#                 kt.Objective('val_recall', direction='max')         # Secondary: Recall
#             ],
#             max_trials=20,
#             num_initial_points=4,
#             directory=fn_args.working_dir,
#             project_name='advanced_medical_tuning',
#             seed=42,
#             overwrite=True
#         )
        
#         print("✓ Advanced multi-objective tuner initialized")
        
#         # Enhanced callbacks dengan medical-specific monitoring
#         def get_advanced_callbacks():
#             return [
#                 # Medical-specific early stopping
#                 keras.callbacks.EarlyStopping(
#                     monitor='val_auc',
#                     patience=6,
#                     restore_best_weights=True,
#                     verbose=1,
#                     mode='max',
#                     min_delta=0.001  # Minimal improvement
#                 ),
                
#                 # Precision-Recall monitoring
#                 keras.callbacks.EarlyStopping(
#                     monitor='val_precision',
#                     patience=8,
#                     restore_best_weights=True,
#                     verbose=0,
#                     mode='max',
#                     min_delta=0.005
#                 ),
                
#                 # Learning rate scheduling
#                 keras.callbacks.ReduceLROnPlateau(
#                     monitor='val_loss',
#                     factor=0.8,
#                     patience=4,
#                     verbose=1,
#                     min_lr=1e-7,
#                     cooldown=2
#                 ),
                
#                 # Custom callback untuk medical metrics
#                 MedicalMetricsCallback()
#             ]
        
#         from tfx.components.tuner.component import TunerFnResult
        
#         return TunerFnResult(
#             tuner=tuner,
#             fit_kwargs={
#                 'x': train_dataset,
#                 'validation_data': eval_dataset,
#                 'steps_per_epoch': 40,  # Efficient tuning
#                 'validation_steps': 20,
#                 'epochs': 20,
#                 'callbacks': get_advanced_callbacks(),
#                 'verbose': 1
#             }
#         )
        
#     except Exception as e:
#         print(f"❌ Error in advanced tuner: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         raise

# Custom callback untuk medical-specific metrics
class MedicalMetricsCallback(keras.callbacks.Callback):
    """Custom callback untuk monitoring medical diagnosis metrics."""
    
    def __init__(self):
        super().__init__()
        self.best_f1 = 0.0
        self.best_balance = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        # Calculate F1 score
        precision = logs.get('val_precision', 0)
        recall = logs.get('val_recall', 0)
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            logs['val_f1_score'] = f1_score
            
            if f1_score > self.best_f1:
                self.best_f1 = f1_score
                print(f"\n🎯 New best F1 score: {f1_score:.4f}")
        
        # Calculate balanced accuracy approximation
        accuracy = logs.get('val_accuracy', 0)
        auc = logs.get('val_auc', 0)
        balanced_score = (accuracy + auc) / 2
        
        if balanced_score > self.best_balance:
            self.best_balance = balanced_score
            print(f"⚖️  New best balanced score: {balanced_score:.4f}")

# Hyperparameter analysis utilities
def analyze_tuning_results(tuner_output_path, top_n=5):
    """Analyze dan extract best hyperparameters dari tuning results."""
    
    try:
        # Load tuner results
        tuner = kt.BayesianOptimization(
            hypermodel=lambda hp: None,  # Dummy hypermodel
            objective='val_auc',
            directory=tuner_output_path,
            project_name='medical_diagnosis_tuning'
        )
        
        # Get best trials
        best_trials = tuner.oracle.get_best_trials(num_trials=top_n)
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} HYPERPARAMETER COMBINATIONS")
        print(f"{'='*60}")
        
        results = []
        for i, trial in enumerate(best_trials):
            print(f"\n🏆 TRIAL {i+1} (Score: {trial.score:.4f})")
            print("-" * 40)
            
            # Extract hyperparameters
            hparams = trial.hyperparameters
            trial_result = {
                'trial_id': trial.trial_id,
                'score': trial.score,
                'hyperparameters': {}
            }
            
            # Kategorisasi hyperparameters
            architecture_params = {}
            training_params = {}
            regularization_params = {}
            
            for param_name in hparams.space:
                param_value = hparams.get(param_name)
                trial_result['hyperparameters'][param_name] = param_value
                
                if any(keyword in param_name.lower() for keyword in ['units', 'layers', 'embed', 'final']):
                    architecture_params[param_name] = param_value
                elif any(keyword in param_name.lower() for keyword in ['learning', 'optimizer', 'weight_decay']):
                    training_params[param_name] = param_value
                elif any(keyword in param_name.lower() for keyword in ['dropout', 'l2', 'batch_norm']):
                    regularization_params[param_name] = param_value
            
            # Print categorized parameters
            if architecture_params:
                print("🏗️  Architecture:")
                for param, value in architecture_params.items():
                    print(f"   {param}: {value}")
            
            if training_params:
                print("🎯 Training:")
                for param, value in training_params.items():
                    print(f"   {param}: {value}")
                    
            if regularization_params:
                print("🛡️  Regularization:")
                for param, value in regularization_params.items():
                    print(f"   {param}: {value}")
            
            results.append(trial_result)
        
        print(f"\n{'='*60}")
        print("HYPERPARAMETER INSIGHTS")
        print(f"{'='*60}")
        
        # Analyze patterns
        if results:
            # Best learning rates
            learning_rates = [r['hyperparameters'].get('learning_rate', 0) for r in results if 'learning_rate' in r['hyperparameters']]
            if learning_rates:
                print(f"📊 Learning Rate Range: {min(learning_rates):.2e} - {max(learning_rates):.2e}")
                print(f"📊 Median Learning Rate: {sorted(learning_rates)[len(learning_rates)//2]:.2e}")
            
            # Best architectures
            num_layers = [r['hyperparameters'].get('num_layers', 0) for r in results if 'num_layers' in r['hyperparameters']]
            if num_layers:
                from collections import Counter
                layer_counts = Counter(num_layers)
                print(f"🏗️  Most common layer count: {layer_counts.most_common(1)[0][0]} layers")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return []

# Export best hyperparameters untuk production use
def export_best_hyperparameters(tuning_results, output_file='best_hyperparams.json'):
    """Export best hyperparameters ke JSON file untuk production."""
    
    import json
    
    if not tuning_results:
        print("❌ No tuning results to export")
        return
    
    best_config = tuning_results[0]  # Best trial
    
    export_data = {
        'model_version': '1.0',
        'tuning_date': str(tf.timestamp()),
        'best_score': float(best_config['score']),
        'hyperparameters': best_config['hyperparameters'],
        'model_config': {
            'task_type': 'binary_classification',
            'domain': 'medical_diagnosis',
            'features': {
                'numerical': NUMERICAL_FEATURES,
                'categorical': CATEGORICAL_FEATURES
            }
        }
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ Best hyperparameters exported to: {output_file}")
        print(f"🎯 Best validation AUC: {best_config['score']:.4f}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error exporting hyperparameters: {e}")
        return None
