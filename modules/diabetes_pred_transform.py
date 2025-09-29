import tensorflow as tf
import tensorflow_transform as tft

# Fitur kategorikal
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

# Fitur numerik
NUMERICAL_FEATURES = [
    'Age'
]

# Label (tidak diproses di Transform)
LABEL_KEY = 'Outcome_Variable'

def preprocessing_fn(inputs):
    """TFX Transform preprocessing function"""
    outputs = {}

    print("=== TFX Transform preprocessing_fn ===")
    print(f"Input keys: {list(inputs.keys())}")
    
    # Transform fitur numerik - scale to 0-1
    for key in NUMERICAL_FEATURES:
        if key in inputs:
            print(f"Processing numerical feature: {key}")
            # Cast ke float32 dan scale ke 0-1
            numeric_feature = tf.cast(inputs[key], tf.float32)
            outputs[f'{key}'] = tft.scale_to_0_1(numeric_feature)
            print(f"✓ Created: {key}")
        else:
            print(f"⚠️  Numerical feature '{key}' not found in inputs")

    # Transform fitur kategorikal menjadi indeks integer
    for key in CATEGORICAL_FEATURES:
        if key in inputs:
            print(f"Processing categorical feature: {key}")
            
            # Handle different data types
            feature_values = inputs[key]
            
            # Convert to string if not already
            if feature_values.dtype == tf.string:
                # Jika sudah string, normalize ke lowercase untuk konsistensi
                string_feature = tf.strings.lower(feature_values)
            else:
                # Jika numerik atau boolean, convert ke string
                string_feature = tf.strings.as_string(feature_values)
            
            # Buat vocabulary dan apply ke feature
            # Ini akan membuat vocabulary file dan mengembalikan integer indices
            outputs[f'{key}'] = tft.compute_and_apply_vocabulary(
                string_feature,
                vocab_filename=f'{key}_vocab',  # Explicit vocabulary filename
                default_value=-1,  # Value for OOV (Out of Vocabulary)
                top_k=None,  # Include all unique values
                frequency_threshold=1,  # Minimum frequency to include in vocab
                num_oov_buckets=1  # Number of OOV buckets
            )
            print(f"✓ Created: {key} with vocabulary {key}_vocab")
        else:
            print(f"⚠️  Categorical feature '{key}' not found in inputs")

    # Label tidak ditransform → diteruskan saja
    if LABEL_KEY in inputs:
        outputs[LABEL_KEY] = tf.cast(inputs[LABEL_KEY], tf.int64)
        print(f"✓ Label '{LABEL_KEY}' passed through as int64")
    else:
        print(f"⚠️  Label '{LABEL_KEY}' not found in inputs")

    print(f"Transform output keys: {list(outputs.keys())}")
    print("=== End Transform preprocessing_fn ===")
    
    return outputs
